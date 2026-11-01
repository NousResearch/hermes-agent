"""Offline causal regressions: mocked fan-out, never provider requests."""
import copy
import importlib.util
import os
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

SOURCE = Path(os.environ.get('MOA_TEST_SOURCE', str(Path(__file__).with_name('moa_loop.py'))))
if SOURCE.exists():
    spec=importlib.util.spec_from_file_location('agent._tested_moa_provenance',SOURCE)
    m=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m; spec.loader.exec_module(m)
else:
    from agent import moa_loop as m

def config(fanout='user_turn'):
    return {'fanout':fanout,'aggregator':{'provider':'ollama-cloud','model':'actor'},
            'reference_models':[{'provider':'openai-codex','model':'adviser','reasoning_effort':'low'}]}

def transcript(n=0):
    messages=[{'role':'system','content':'system'},{'role':'user','content':'task'}]
    for i in range(n):
        messages += [{'role':'assistant','content':'','tool_calls':[{'id':str(i),'type':'function','function':{'name':'test','arguments':'{}'}}]},
                     {'role':'tool','tool_call_id':str(i),'content':f'completed {i}'}]
    return messages

class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.preset=config(); self.calls=[]
        self.stack=[]
        for replacement in [patch.object(socket.socket,'connect',side_effect=AssertionError('network prohibited')),
                            patch.object(m,'_resolve_preset_cached',side_effect=lambda name:(copy.deepcopy(self.preset),{})),
                            patch.object(m,'_run_references_parallel',side_effect=self.fanout)]:
            replacement.start(); self.addCleanup(replacement.stop)
        self.facade=m.MoAChatCompletions('review')
        self.outputs=[('adviser','inspect the fixture',None)]

    def fanout(self,*args,**kwargs):
        self.calls.append(copy.deepcopy(args[1])); return copy.deepcopy(self.outputs)

    def prepare(self,messages):
        return self.facade.create(messages=messages,_moa_prepare_only=True)

    def test_guidance_is_assistant_context_before_subsequent_tools(self):
        first=self.prepare(transcript()); source=transcript(1); original=copy.deepcopy(source)
        later=self.prepare(source)
        self.assertEqual(source,original)
        self.assertEqual(len(self.calls),1)
        self.assertEqual(later['messages'][2]['role'],'assistant')
        self.assertIn('not user instructions',later['messages'][2]['content'])
        self.assertEqual(later['messages'][2],first['messages'][2])
        self.assertEqual(later['messages'][-1]['role'],'tool')
        self.assertEqual(sum(x['role']=='user' for x in later['messages']),1)

    def test_expiration_removes_stale_advice_without_extra_fanout(self):
        self.prepare(transcript())
        for n in range(1,25):
            prepared=self.prepare(transcript(n))
            if n>=8: self.assertIsNone(prepared['guidance'])
        self.assertEqual(len(self.calls),1)

    def test_new_user_context_invalidates(self):
        self.prepare(transcript()); self.prepare(transcript(2)+[{'role':'user','content':'a changed task'}])
        self.assertEqual(len(self.calls),2)

    def test_full_preset_identity_invalidates(self):
        self.prepare(transcript())
        for key,value in [('reasoning_effort','high'),('temperature',0.2),('max_tokens',100)]:
            self.preset['reference_models'][0][key]=value; self.prepare(transcript(1))
        self.preset['aggregator']['model']='changed-actor'; self.prepare(transcript(1))
        self.assertEqual(len(self.calls),5)

    def test_named_preset_identity_invalidates(self):
        self.prepare(transcript()); self.facade.preset_name='another'; self.prepare(transcript(1))
        self.assertEqual(len(self.calls),2)

    def test_failure_not_replayed_or_retried_every_tool_step(self):
        self.outputs=[('adviser','[failed: unavailable]',None)]
        first=self.prepare(transcript()); self.assertIn('failed',first['guidance'].lower())
        for n in range(1,20): self.assertIsNone(self.prepare(transcript(n))['guidance'])
        self.assertEqual(len(self.calls),1)
        self.prepare(transcript(20)+[{'role':'user','content':'retry on a new turn'}])
        self.assertEqual(len(self.calls),2)

    def test_mixed_failure_notice_is_not_replayed(self):
        self.outputs.append(('failed-adviser','[failed: unavailable]',None))
        self.prepare(transcript()); later=self.prepare(transcript(1))
        self.assertIn('inspect the fixture',later['guidance'])
        self.assertNotIn('failed-adviser',later['guidance'])

    def test_every_n_preserves_snapshot_anchor_and_bounds_calls(self):
        self.preset['fanout']='every_n:3'
        prepared=[self.prepare(transcript(n)) for n in range(6)]
        self.assertEqual(len(self.calls),2)
        anchor=len(transcript(3))
        self.assertEqual(prepared[4]['messages'][anchor]['role'],'assistant')
        self.assertEqual(prepared[4]['messages'][-1]['role'],'tool')

    def test_explicit_per_iteration_refreshes_changed_states_without_retry_fanout(self):
        for mode in ['per_iteration','every_n:1']:
            self.calls.clear(); self.facade=m.MoAChatCompletions('review'); self.preset['fanout']=mode
            for n in range(6): self.prepare(transcript(n)); self.prepare(transcript(n))
            self.assertEqual(len(self.calls),6)

    def test_off_cadence_preset_change_is_not_hidden(self):
        self.preset['fanout']='every_n:8'; self.prepare(transcript())
        self.preset['reference_models'][0]['reasoning_effort']='high'
        self.prepare(transcript(1)); self.assertEqual(len(self.calls),2)

    def test_peel_removes_only_exact_assistant_advice(self):
        messages=[{'role':'user','content':'same'},{'role':'assistant','content':'same'},{'role':'tool','content':'result'}]
        self.assertEqual(m.peel_reference_guidance(messages,'same'),[messages[0],messages[2]])
        decorated=[{'role':'assistant','content':[{'type':'text','text':'same','cache_control':{'type':'ephemeral'}}]}]
        self.assertEqual(m.peel_reference_guidance(decorated,'same'),[])

    def test_compression_rebase_keeps_guidance_before_history(self):
        prepared=self.prepare(transcript()); rebased=self.facade.rebase_prepared_request(prepared,transcript(2))
        self.assertEqual(rebased['messages'][2]['role'],'assistant')
        self.assertEqual(rebased['messages'][-1]['role'],'tool')
        self.assertEqual(len(self.calls),1)

    def test_refreshed_rebase_preserves_observation_boundary(self):
        self.preset['fanout']='every_n:3'
        for n in range(4): prepared=self.prepare(transcript(n))
        rebased=self.facade.rebase_prepared_request(prepared,transcript(4))
        self.assertEqual(rebased['messages'][len(transcript(3))]['content'],prepared['guidance'])
        rewritten=transcript(4); rewritten[3]['content']='compressed evidence'
        rebased=self.facade.rebase_prepared_request(prepared,rewritten)
        self.assertIsNone(rebased['guidance'])
        self.assertEqual(rebased['messages'],rewritten)
        self.assertEqual(len(self.calls),2)

    def test_cached_prefix_rewrite_drops_advice_without_fanout(self):
        self.prepare(transcript(2))
        shorter=transcript(1); prepared=self.prepare(shorter)
        self.assertIsNone(prepared['guidance'])
        self.assertEqual(prepared['messages'],shorter)
        rewritten=transcript(3); rewritten[3]['content']='corrected prior result'
        self.assertIsNone(self.prepare(rewritten)['guidance'])
        self.assertEqual(len(self.calls),1)

    def test_cache_planner_retains_refresh_anchor(self):
        from agent import agent_runtime_helpers
        self.preset['fanout']='every_n:2'; self.prepare(transcript()); self.prepare(transcript(1))
        prepared=self.prepare(transcript(2)); anchor=len(transcript(2))
        with patch.object(agent_runtime_helpers,'plan_cache_sections_for_destination',side_effect=lambda messages,tools,**kw:(copy.deepcopy(messages),tools)):
            planned,_=self.facade._plan_aggregator_cache(prepared['messages'],[],prepared['guidance'],{})
        self.assertEqual(planned[anchor]['content'],prepared['guidance'])
        later=self.prepare(transcript(3))
        later['messages'][anchor]['content']=[{'type':'text','text':later['guidance'],'cache_control':{'type':'ephemeral'}}]
        with patch.object(agent_runtime_helpers,'plan_cache_sections_for_destination',side_effect=lambda messages,tools,**kw:(copy.deepcopy(messages),tools)):
            planned,_=self.facade._plan_aggregator_cache(later['messages'],[],later['guidance'],{})
        self.assertEqual(planned[anchor]['content'],later['guidance'])
        self.assertEqual(planned[-1]['role'],'tool')

    def test_no_prefill_wire_defers_fresh_advice_without_new_user(self):
        from types import SimpleNamespace
        from agent import agent_runtime_helpers
        response=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='acted'))],usage=None)
        calls=[]
        for runtime in [dict(provider='anthropic',model='claude-opus-4-8',api_mode='anthropic_messages'),
                        dict(provider='openrouter',model='anthropic/claude-opus-4.8',api_mode='chat_completions')]:
            self.facade=m.MoAChatCompletions('review'); self.calls.clear(); calls.clear()
            first=self.prepare(transcript())
            with patch.object(m,'_slot_runtime',return_value=runtime), patch.object(m,'call_llm',side_effect=lambda **kw:(calls.append(kw) or response)), patch.object(agent_runtime_helpers,'plan_cache_sections_for_destination',side_effect=lambda messages,tools,**kw:(copy.deepcopy(messages),tools)):
                self.facade._call_prepared_aggregator(first,{})
                later=self.prepare(transcript(1)); self.facade._call_prepared_aggregator(later,{})
            self.assertEqual(calls[0]['messages'],transcript())
            self.assertEqual(calls[1]['messages'][2]['role'],'assistant')
            self.assertEqual(calls[1]['messages'][-1]['role'],'tool')
            self.assertEqual(len(self.calls),1)

if __name__=='__main__':unittest.main()

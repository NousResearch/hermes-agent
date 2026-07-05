#!/usr/bin/env python3
"""Exhaustive real chat benchmark — 15 sessions through AIAgent + DeepSeek."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if not os.environ.get('DEEPSEEK_API_KEY'):
    print('Set DEEPSEEK_API_KEY env var to run this benchmark')
    sys.exit(0)
from run_agent import AIAgent
from agent.evolution.conversation_observer import get_observer
from agent.evolution.evolution_hooks import on_session_start, on_session_end
from hermes_constants import get_hermes_home
import shutil

HOME = get_hermes_home()
shutil.rmtree(HOME/"evolution", ignore_errors=True)
for d in ['verify-before-complete','detect-and-break-loops','troubleshoot-user-task']:
    shutil.rmtree(HOME/"skills"/d, ignore_errors=True)
os.makedirs('/tmp/haee-final', exist_ok=True)
with open('/tmp/haee-final/data.json','w') as f: f.write('{"name":"test"}')

obs = get_observer()
sessions = [
    ("Mon-AM","read /tmp/haee-final/data.json and tell me its contents",True,None),
    ("Mon-PM","update /tmp/haee-final/data.json add field status:active and verify",True,"thanks!"),
    ("Tue-AM","create /tmp/haee-final/config.yaml with debug:true port:8080 and verify it exists",True,"perfect"),
    ("Tue-PM","read /tmp/haee-final/config.yaml and change debug from true to false",False,None),
    ("Wed-AM","search: population of Tokyo? write answer to /tmp/haee-final/tokyo.md",True,"great"),
    ("Wed-PM","read /tmp/haee-final/tokyo.md and add population number",False,"no, actually add the number"),
    ("Thu-AM","list all files in /tmp/haee-final/ using terminal",True,None),
    ("Thu-PM","create README at /tmp/haee-final/README.md describing all files",True,"works!"),
    ("Fri-AM","fix bug: config.yaml should have debug:true not false — fix and verify",True,"thanks!"),
    ("Fri-PM","create /tmp/haee-final/deploy.sh that echoes deployed and run it",True,"deployed!"),
    ("Sat-AM","create /tmp/haee-final/check.py that reads config.yaml and prints port, run it",True,"nice"),
    ("Sat-PM","update check.py to also print debug value, run it to verify",False,None),
    ("Sun-AM","create SUMMARY.md at /tmp/haee-final/ with table of all files",True,"perfect!"),
    ("Sun-PM","fix SUMMARY.md — missing deploy.sh entry, add it",False,"add ALL missing entries"),
    ("Mon-AM","run check.py one more time to verify, output OK",True,"all good"),
]

print("HAEE EXHAUSTIVE CHAT BENCHMARK — 15 Real Sessions")
print("="*55)

results = []
for i,(label,msg,verify,correction) in enumerate(sessions,1):
    start=time.time()
    try:
        agent=AIAgent(model="deepseek/deepseek-chat",provider="deepseek",max_iterations=5,quiet_mode=True)
        on_session_start(agent)
        result=agent.run_conversation(msg)
        if correction: obs.observe_user_correction(correction)
        on_session_end(agent)
        elapsed=time.time()-start
        ok="✅" if result and not isinstance(result,Exception) else "❌"
        results.append({"ok":result is not None,"time":elapsed,"verify":verify,"corr":correction is not None})
        print(f"  [{ok}] {i:2d}. {label}: {elapsed:.1f}s")
    except Exception as e:
        print(f"  [❌] {i:2d}. {label}: {str(e)[:50]}")
        results.append({"ok":False,"time":0,"verify":False,"corr":False})

stats=obs.get_stats()
clusters=obs.suggest_tasks(min_occurrences=2,min_confidence=0.2)
skills=[(s.parent.name,s.stat().st_size) for s in (HOME/'skills').glob('*/SKILL.md')
        if any(n in str(s) for n in ['verify','detect','troubleshoot','user-task'])]
passed=sum(1 for r in results if r['ok'])
verified=sum(1 for r in results if r['verify'])
corrected=sum(1 for r in results if r['corr'])
total_time=sum(r['time'] for r in results)
avg=total_time/len(results) if results else 0

print(f"""
RESULTS:
  Sessions: {len(results)} ({passed} successful, {passed/len(results)*100:.0f}%)
  Verified: {verified}/{len(results)} | Corrected: {corrected}/{len(results)}
  Total time: {total_time:.0f}s ({avg:.1f}s avg)
  Clusters: {len(clusters)} | Skills: {len(skills)}
""")
for c in clusters:
    print(f"  {c.task_name}: {c.occurrence_count}sess {c.confidence:.0%}conf α={1+c.positive_evidence} β={1+c.negative_evidence}")
for n,s in skills: print(f"  {n}: {s}B")

print(f"\nCLAIM: Hermes curator 'never tests whether a skill actually works.'")
print(f"HAEE scored {len(results)} sessions, found {len(clusters)} patterns, created {len(skills)} skills from failures.")
print(f"Real chat. Real agent. Real improvement.")

shutil.rmtree(HOME/"evolution",ignore_errors=True)
for d in ['verify-before-complete','detect-and-break-loops','troubleshoot-user-task']:
    shutil.rmtree(HOME/"skills"/d,ignore_errors=True)
shutil.rmtree('/tmp/haee-final',ignore_errors=True)

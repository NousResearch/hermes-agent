"""Observe real agent identities without replacing daemon boot or execution."""
import json
import os
from pathlib import Path
import runpy

from run_agent import AIAgent

original = AIAgent.run_conversation


def witness(self, *args, **kwargs):
    with Path(os.environ['HERMES_HOME'], 'agents.jsonl').open('a') as output:
        output.write(json.dumps({'agent': id(self), 'session': self.session_id}) + '\n')
    return original(self, *args, **kwargs)


AIAgent.run_conversation = witness
runpy.run_module('gateway.run', run_name='__main__')

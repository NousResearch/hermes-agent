"""#103492: a hardline command hidden after a NEWLINE inside a double-quoted $(grep …) must stay blocked
by the public guard with no approval callback; a grep with a backtick operand must stay allowed.

Independent-review probe (written by the /review subagent for tracking issue #103563, adapted). It
reproduced an approval bypass in the first version of the PR (approved, 0 callbacks). Runs the real
public guard (check_dangerous_command) in a temp HERMES_HOME with approvals.mode=manual and executes a
HARMLESS Bash witness (reboot shadowed by a function writing a marker) to prove reachability.

Usage: python scanner_bypass_probe.py <repo_root> [<baseline_approval_detection.py>]
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
ROOT=Path(sys.argv[1]).resolve()  # repo root under test
sys.path.insert(0,str(ROOT))
home=Path(tempfile.mkdtemp(prefix='review103492-confirm-'))
os.environ['HERMES_HOME']=str(home)
os.environ['HERMES_INTERACTIVE']='1'
(home/'config.yaml').write_text('approvals:\n  mode: manual\n  timeout: 1\n', encoding='utf-8')
from tools import approval_detection as d
from tools import approval as a
b=None
if len(sys.argv)>2:  # optional: path to a baseline approval_detection.py (e.g. from main) to compare verdicts
    spec=importlib.util.spec_from_file_location('baseline',sys.argv[2]); b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
r='re'+'boot'
cmd='echo "$(grep -P \'safe\' /dev/null\n'+r+')"'
print(json.dumps({'module':d.__file__,'approval_module':a.__file__,'case':'newline command','command':cmd,'baseline':(b.detect_hardline_command(cmd) if b else None),'head':d.detect_hardline_command(cmd),'variants':list(d._command_detection_variants(cmd))}))
calls=[]
def deny(*args,**kwargs):
    calls.append([args,kwargs]);return 'deny'
print(json.dumps({'pattern_guard':a.check_dangerous_command(cmd,'local',approval_callback=deny),'combined_guard':a.check_all_command_guards(cmd,'local',approval_callback=deny),'callback_calls':calls},default=str))
# Shadow the dangerous command name with a harmless function; verify shell
# execution and marker contents, never invoke a host shutdown binary.
marker=home/'marker'
prefix=r+'(){ printf executed > "$MARKER"; }; '
p=subprocess.run(['/bin/bash','--noprofile','--norc','-c',prefix+cmd],env={'PATH':'/usr/bin:/bin','HOME':str(home),'MARKER':str(marker)},capture_output=True,text=True, encoding='utf-8', errors='replace',timeout=5)
assert marker.read_text(encoding='utf-8')=='executed'
print(json.dumps({'case':'newline safe execution','exit':p.returncode,'marker':marker.read_text(encoding='utf-8'),'stdout':p.stdout,'stderr':p.stderr}))
# Prove the benign backtick argument really is well-formed and matches input.
f=home/'f';f.write_text('needle\n', encoding='utf-8')
cmd='grep -e `printf needle` '+str(f)
p=subprocess.run(['/bin/bash','--noprofile','--norc','-c',cmd],env={'PATH':'/usr/bin:/bin','HOME':str(home)},capture_output=True,text=True, encoding='utf-8', errors='replace',timeout=5)
assert p.returncode==0 and p.stdout=='needle\n'
print(json.dumps({'case':'backtick argument','command':cmd,'baseline':(b.detect_hardline_command(cmd) if b else None),'head':d.detect_hardline_command(cmd),'tokens':d._shell_tokens_with_spans(cmd,0),'exit':p.returncode,'stdout':p.stdout}))
# Reporter's exact spelling, fixture makes the sed address meaningful.
with tempfile.TemporaryDirectory() as tmp:
    Path(tmp,'f').write_text('X\ny\nz\nw\n', encoding='utf-8')
    cmd='sed -n "$(grep -n X f | cut -d: -f1),+3p" f'
    p=subprocess.run(['/bin/bash','--noprofile','--norc','-c',cmd],cwd=tmp,env={'PATH':'/usr/bin:/bin','HOME':str(home)},capture_output=True,text=True, encoding='utf-8', errors='replace',timeout=5)
    assert p.returncode==0 and p.stdout=='X\ny\nz\nw\n'
    print(json.dumps({'case':'reported real fixture','baseline':(b.detect_hardline_command(cmd) if b else None),'head':d.detect_hardline_command(cmd),'exit':p.returncode,'stdout':p.stdout}))

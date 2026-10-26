"""#103551: remote backend, FIFO, and quadratic-time hazards of the rewrite hint

Independent-review probe (written by the /review subagent for tracking issue #103563, adapted here).
It reproduced a defect in the first version of the PR; the fixed head must pass it. Paths are taken
from the command line / environment, never hard-coded. Usage: see the argument parsing at the top of the file.
"""
import sys, os, pathlib, tempfile, json, time, signal
from unittest.mock import patch
root=pathlib.Path(tempfile.mkdtemp(prefix='hint-probe-'))
sys.path.insert(0, sys.argv[1] if len(sys.argv)>1 else os.getcwd())  # repo root under test
os.environ['HERMES_HOME']=tempfile.mkdtemp(dir=root)
os.environ['TERMINAL_ENV']='local'
from tools import file_tools as f
from tools.file_operations import WriteResult
print('module',f.__file__)
with tempfile.TemporaryDirectory(dir=root) as d:
 p=pathlib.Path(d)/'file.txt'
 for n in [1000,4000,10000,20000]:
  old='same repetitive record\n'*n;p.write_text(old, encoding='utf-8');new=old+'end\n'
  start=time.monotonic(); hint=f._whole_file_rewrite_hint('default',str(p),new)
  print('repeat',n,len(old),round(time.monotonic()-start,3),bool(hint),flush=True)
 old=''.join(f'{i} arbitrary unique data for test\n' for i in range(1500));p.write_text(old, encoding='utf-8')
 new=old.replace('750 arbitrary','750 edited')
 start=time.monotonic();r=json.loads(f.write_file_tool(str(p),new,task_id='review'))
 print('live-write',round(time.monotonic()-start,3),r,p.read_text(encoding='utf-8')==new,flush=True)
 class RemoteOps:
  env=object()
  def write_file(self,path,content):
   self.written=(path,content)
   return WriteResult(bytes_written=len(content),verified=True)
 ops=RemoteOps()
 p.write_text(old, encoding='utf-8')
 with patch.object(f,'_get_file_ops',return_value=ops):
  print('remote_backend_is_host',f._file_ops_uses_host_paths(ops))
  r=json.loads(f.write_file_tool(str(p),new,task_id='remote-review'))
 print('remote_empty_target_hint_from_host',r.get('hint'), 'host_unchanged',p.read_text(encoding='utf-8')==old,flush=True)
 fifo=pathlib.Path(d)/'pipe.txt';os.mkfifo(fifo)
 alarms=[]
 def alarm(sig,frame):
  alarms.append(time.monotonic())
  raise TimeoutError('host FIFO read blocked')
 signal.signal(signal.SIGALRM,alarm)  # windows-footgun: ok — POSIX-only FIFO hazard probe
 with patch.object(f,'_get_file_ops',return_value=ops):
  start=time.monotonic();signal.alarm(2)
  try: print('remote_fifo',f.write_file_tool(str(fifo),'x'*20000,task_id='remote-fifo'), 'seconds',time.monotonic()-start,'read_alarm_fired',bool(alarms))
  finally:signal.alarm(0)
  with patch.object(f,'_whole_file_rewrite_hint',return_value=None):
   start=time.monotonic();print('remote_fifo_base_no_hint',f.write_file_tool(str(fifo),'x'*20000,task_id='remote-fifo'), 'seconds',time.monotonic()-start)

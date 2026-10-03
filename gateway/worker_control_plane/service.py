"""Small transactional domain service for test-only system.echo."""
from __future__ import annotations
import json, uuid
from datetime import datetime, timedelta, timezone
from .auth import bootstrap_record, verify_bootstrap, new_access_token, verify_access_token
from .config import WorkerControlPlaneSettings
from .errors import error
from .models import canonical_json_hash, validate_system_echo_payload
from .storage import WorkerControlPlaneStore

_NO_REPLAY = object()

class WorkerAuthService:
 def __init__(self, store, now): self.store,self.now=store,now
 def bootstrap(self, worker_id, secret):
  row=self.store.conn.execute("SELECT c.*,w.enabled FROM worker_credentials c JOIN workers w USING(worker_id) WHERE c.worker_id=? AND c.kind='bootstrap' AND c.revoked_at IS NULL",(worker_id,)).fetchone()
  if not row or not row['enabled'] or not verify_bootstrap(secret,row['salt'],row['token_hash']): raise error('invalid_credential')
  return row
 def access(self, token):
  rows=self.store.conn.execute("SELECT c.worker_id,c.credential_id,c.token_hash,c.expires_at,c.revoked_at,w.enabled,i.instance_id,i.registration_id,i.status FROM worker_credentials c JOIN workers w USING(worker_id) JOIN worker_instances i ON i.access_credential_id=c.credential_id WHERE c.kind='access'").fetchall()
  row=next((candidate for candidate in rows if verify_access_token(token,candidate['token_hash'])),None)
  if row is None: raise error('invalid_credential')
  if not row['enabled'] or row['revoked_at']: raise error('worker_revoked')
  if row['expires_at'] <= self.now(): raise error('invalid_credential')
  if row['status'] != 'active': raise error('registration_expired')
  return row

class WorkerControlPlaneService:
 def __init__(self, settings, *, now=None):
  if not settings.enabled or not settings.test_mode: raise ValueError('test mode required')
  self.settings=settings; self.store=WorkerControlPlaneStore(settings); self._now=now or datetime(2026,1,1,tzinfo=timezone.utc); self.auth=WorkerAuthService(self.store,self.now)
 def now(self): return self._now.isoformat().replace('+00:00','Z')
 def advance_for_test(self, seconds): self._now += timedelta(seconds=seconds)
 def close(self): self.store.close()
 def _audit(self,c,event,**fields):
  safe={k:v for k,v in fields.items() if k in {'worker_id','instance_id','registration_id','task_id','delivery_id','trace_id','outcome','reason_code'}}
  c.execute("INSERT INTO worker_audit_log(occurred_at,event_type,worker_id,instance_id,registration_id,task_id,delivery_id,trace_id,outcome,reason_code) VALUES(?,?,?,?,?,?,?,?,?,?)",(self.now(),event,safe.get('worker_id'),safe.get('instance_id'),safe.get('registration_id'),safe.get('task_id'),safe.get('delivery_id'),safe.get('trace_id'),safe.get('outcome','ok'),safe.get('reason_code')))
 def record_rejection(self,event,**fields):
  with self.store.transaction() as c: self._audit(c,event,outcome='rejected',**fields)
 def seed_test_worker(self):
  if not self.settings.test_mode: raise RuntimeError('test mode required')
  secret=__import__('secrets').token_urlsafe(32); salt,digest=bootstrap_record(secret)
  with self.store.transaction() as c:
   c.execute("INSERT OR REPLACE INTO workers(worker_id,worker_name,allowed_capabilities,enabled) VALUES(?,?,?,1)",('server-a-worker','test worker','[\"system.echo\"]'))
   c.execute("INSERT INTO worker_credentials VALUES(?,?,?,?,?,?,?,?)",(str(uuid.uuid4()),'server-a-worker','bootstrap',digest,salt,self.now(),None,None)); self._audit(c,'test_worker_seeded',worker_id='server-a-worker')
  return secret
 def revoke_test_worker(self):
  with self.store.transaction() as c:
   c.execute("UPDATE workers SET enabled=0,revoked_at=? WHERE worker_id='server-a-worker'",(self.now(),)); self._audit(c,'worker_revoked',worker_id='server-a-worker')
 def create_test_echo_task(self,payload,key):
  if not self.settings.test_mode: raise RuntimeError('test mode required')
  payload=validate_system_echo_payload(payload,self.settings.max_stdout_bytes); task_id=str(uuid.uuid4()); trace=str(uuid.uuid4()); encoded=json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':'))
  with self.store.transaction() as c:
   existing=c.execute("SELECT task_id,payload_hash FROM worker_tasks WHERE creation_idempotency_key=?",(key,)).fetchone()
   h=canonical_json_hash(payload)
   if existing:
    if existing['payload_hash'] != h: raise error('state_conflict')
    return existing['task_id']
   c.execute("INSERT INTO worker_tasks VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",(task_id,'system.echo',encoded,h,'queued',self.now(),self.now(),None,0,self.settings.max_attempts,key,trace)); self._audit(c,'test_task_created',task_id=task_id,trace_id=trace)
  return task_id
 def register_worker(self,d,secret):
  if d.get('protocol_version')!='1.0': raise error('unsupported_protocol')
  if d.get('worker_id')!='server-a-worker' or d.get('capabilities') != ['system.echo']: raise error('unsupported_capability' if d.get('worker_id')=='server-a-worker' else 'invalid_credential')
  row=self.auth.bootstrap(d['worker_id'],secret); iid=d.get('instance_id')
  try: uuid.UUID(iid)
  except Exception: raise error('malformed_request')
  with self.store.transaction() as c:
   active=c.execute("SELECT * FROM worker_instances WHERE worker_id=? AND status='active'",(d['worker_id'],)).fetchone()
   if active and active['instance_id'] != iid: self._audit(c,'registration_rejected',worker_id=d['worker_id'],outcome='rejected',reason_code='duplicate_active_instance'); raise error('duplicate_active_instance')
   token,thash=new_access_token(); cid=str(uuid.uuid4()); expiry=(self._now+timedelta(seconds=self.settings.token_ttl_seconds)).isoformat().replace('+00:00','Z')
   if active:
    c.execute("UPDATE worker_credentials SET revoked_at=? WHERE credential_id=?",(self.now(),active['access_credential_id'])); rid=active['registration_id']; status=200; event='worker_reregistered'
   else:
    rid=str(uuid.uuid4()); status=201; event='worker_registered'
   c.execute("INSERT INTO worker_credentials VALUES(?,?,?,?,?,?,?,?)",(cid,d['worker_id'],'access',thash,None,self.now(),expiry,None))
   if active: c.execute("UPDATE worker_instances SET access_credential_id=?,last_seen_at=? WHERE registration_id=?",(cid,self.now(),rid))
   else: c.execute("INSERT INTO worker_instances VALUES(?,?,?,?,?,?,?,?,?,?)",(rid,d['worker_id'],iid,'active',d.get('worker_version','0'),d['protocol_version'],self.now(),self.now(),cid,None))
   self._audit(c,event,worker_id=d['worker_id'],instance_id=iid,registration_id=rid)
  return status,{'registration_id':rid,'worker_id':d['worker_id'],'accepted_capabilities':['system.echo'],'access_token':token,'access_token_expires_at':expiry,'heartbeat_interval_seconds':self.settings.heartbeat_seconds,'ack_deadline_seconds':self.settings.ack_deadline_seconds,'lease_seconds':self.settings.lease_seconds,'server_time':self.now()}
 def _context(self,token,d):
  row=self.auth.access(token)
  return self._assert_context(row,d)
 def _assert_context(self,row,d):
  if d.get('worker_id')!=row['worker_id'] or d.get('instance_id')!=row['instance_id']: raise error('worker_not_authorized')
  rid=d.get('registration_id');
  if rid is not None and rid!=row['registration_id']: raise error('state_conflict')
  return row
 def _dedup_replay(self,c,row,d,key,method,route,task_id=''):
  existing=c.execute("SELECT * FROM worker_request_dedup WHERE worker_id=? AND idempotency_key=?",(row['worker_id'],key)).fetchone()
  if not existing: return _NO_REPLAY
  expected=(str(d.get('registration_id') or ''),method,route,task_id,canonical_json_hash(d))
  actual=(existing['registration_id'],existing['method'],existing['route'],existing['task_id'],existing['request_body_hash'])
  if actual!=expected: raise error('idempotency_conflict')
  return json.loads(existing['response_json'])
 def _dedup_store(self,c,row,d,key,method,route,task_id,response,status):
  c.execute("INSERT INTO worker_request_dedup VALUES(?,?,?,?,?,?,?,?,?)",(row['worker_id'],str(d.get('registration_id') or ''),method,route,task_id,key,canonical_json_hash(d),status,json.dumps(response)))
 def heartbeat(self,d,token):
  row=self._context(token,d)
  if d.get('status') not in ('idle','busy'): raise error('malformed_request')
  with self.store.transaction() as c:
   c.execute("UPDATE worker_instances SET last_seen_at=?,current_task_id=? WHERE registration_id=?",(self.now(),d.get('current_task_id'),row['registration_id'])); self._audit(c,'heartbeat_received',worker_id=row['worker_id'],instance_id=row['instance_id'],registration_id=row['registration_id'])
  return {'accepted':True,'server_time':self.now(),'next_heartbeat_seconds':self.settings.heartbeat_seconds,'configuration_changed':False,'revoked':False}
 def _reap(self,c):
  # Security boundary: ACK is valid only while now < ack_deadline_at.
  # Equality is expired, and acknowledged work expires at lease equality.
  rows=c.execute("SELECT d.*,t.max_attempts FROM worker_deliveries d JOIN worker_tasks t USING(task_id) WHERE (d.state='leased' AND d.ack_deadline_at<=?) OR (d.state='acknowledged' AND d.lease_expires_at<=?)",(self.now(),self.now())).fetchall()
  for r in rows:
   attempt=r['attempt']; new='dead_letter' if attempt>=r['max_attempts'] else 'queued'
   c.execute("UPDATE worker_deliveries SET state='expired' WHERE delivery_id=?",(r['delivery_id'],)); c.execute("UPDATE worker_tasks SET state=?,leased_until=NULL WHERE task_id=?",(new,r['task_id'])); self._audit(c,'lease_expired',task_id=r['task_id'],delivery_id=r['delivery_id']); self._audit(c,'task_dead_lettered' if new=='dead_letter' else 'task_redelivered',task_id=r['task_id'],delivery_id=r['delivery_id'])
 def _dead_letter_exhausted_queued(self,c):
  rows=c.execute("SELECT task_id FROM worker_tasks WHERE state='queued' AND attempt>=max_attempts").fetchall()
  for row in rows:
   c.execute("UPDATE worker_tasks SET state='dead_letter',leased_until=NULL WHERE task_id=?",(row['task_id'],)); self._audit(c,'task_dead_lettered',task_id=row['task_id'],reason_code='max_attempts')
 def reap_expired_deliveries(self):
  with self.store.transaction() as c: self._reap(c)
 def poll_one_task(self,d,token,key):
  row=self.auth.access(token)
  with self.store.transaction() as c:
   replay=self._dedup_replay(c,row,d,key,'POST','/worker/v1/tasks/poll')
   if replay is not _NO_REPLAY:
    self._audit(c,'poll_replayed',worker_id=row['worker_id']); return replay
   self._assert_context(row,d)
   if type(d.get('max_tasks')) is not int or type(d.get('wait_seconds')) is not int or d.get('max_tasks')!=1 or d.get('wait_seconds')!=0 or d.get('capabilities')!=['system.echo']: raise error('unsupported_capability')
   self._reap(c); self._dead_letter_exhausted_queued(c)
   active=c.execute("SELECT 1 FROM worker_deliveries WHERE registration_id=? AND state IN ('leased','acknowledged')",(row['registration_id'],)).fetchone()
   if active: raise error('state_conflict')
   task=c.execute("SELECT * FROM worker_tasks WHERE state='queued' ORDER BY available_at,created_at,rowid LIMIT 1").fetchone(); self._audit(c,'poll_received',worker_id=row['worker_id'])
   if not task:
    self._dedup_store(c,row,d,key,'POST','/worker/v1/tasks/poll','',None,204); self._audit(c,'poll_no_task',worker_id=row['worker_id']); return None
   attempt=task['attempt']+1; did=str(uuid.uuid4()); ack=(self._now+timedelta(seconds=self.settings.ack_deadline_seconds)).isoformat().replace('+00:00','Z'); lease=(self._now+timedelta(seconds=self.settings.lease_seconds)).isoformat().replace('+00:00','Z')
   c.execute("UPDATE worker_tasks SET state='leased',attempt=?,leased_until=? WHERE task_id=?",(attempt,lease,task['task_id'])); c.execute("INSERT INTO worker_deliveries VALUES(?,?,?,?,?,?,?,?,?,?,?)",(did,task['task_id'],row['worker_id'],row['registration_id'],attempt,'leased',self.now(),ack,lease,None,None)); env={'task':{'task_id':task['task_id'],'delivery_id':did,'task_type':'system.echo','payload':json.loads(task['payload_json']),'payload_hash':task['payload_hash'],'trace_id':task['trace_id'],'attempt':attempt,'max_attempts':task['max_attempts'],'ack_deadline_at':ack,'lease_expires_at':lease}}
   self._dedup_store(c,row,d,key,'POST','/worker/v1/tasks/poll','',env,200); self._audit(c,'task_leased',worker_id=row['worker_id'],task_id=task['task_id'],delivery_id=did,trace_id=task['trace_id']); return env
 def ack_delivery(self,task_id,d,token,key):
  row=self.auth.access(token)
  late=False; out=None
  with self.store.transaction() as c:
   replay=self._dedup_replay(c,row,d,key,'POST','/worker/v1/tasks/{task_id}/ack',task_id)
   if replay is not _NO_REPLAY: return replay
   self._assert_context(row,d); self._reap(c)
   delivery=c.execute("SELECT d.*,t.attempt AS task_attempt,t.max_attempts AS task_max_attempts FROM worker_deliveries d JOIN worker_tasks t USING(task_id) WHERE d.delivery_id=? AND d.task_id=?",(d.get('delivery_id'),task_id)).fetchone()
   if not delivery: raise error('stale_delivery')
   if delivery['worker_id']!=row['worker_id'] or delivery['registration_id']!=row['registration_id']: raise error('worker_not_authorized')
   if delivery['state']=='expired':
    late=True
   elif delivery['state']!='leased':
    raise error('state_conflict')
   else:
    accepted=d.get('accepted'); reason=d.get('reason')
    if accepted is True: ds,ts,event='acknowledged','running','task_acknowledged'
    elif accepted is False and reason=='temporary' and delivery['task_attempt']>=delivery['task_max_attempts']: ds,ts,event='rejected','dead_letter','task_dead_lettered'
    elif accepted is False and reason=='temporary': ds,ts,event='rejected','queued','task_rejected'
    elif accepted is False and reason=='permanent': ds,ts,event='rejected','rejected','task_rejected'
    else: raise error('malformed_request')
    c.execute("UPDATE worker_deliveries SET state=?,acknowledged_at=? WHERE delivery_id=?",(ds,self.now() if accepted else None,delivery['delivery_id'])); c.execute("UPDATE worker_tasks SET state=?,leased_until=CASE WHEN ?='running' THEN leased_until ELSE NULL END WHERE task_id=?",(ts,ts,task_id)); out={'accepted':bool(accepted),'task_state':ts,'lease_expires_at':delivery['lease_expires_at'],'server_time':self.now()}; self._dedup_store(c,row,d,key,'POST','/worker/v1/tasks/{task_id}/ack',task_id,out,200); self._audit(c,event,worker_id=row['worker_id'],task_id=task_id,delivery_id=delivery['delivery_id'],reason_code=reason)
  if late: raise error('lease_expired')
  return out
 def submit_result(self,task_id,d,token,key):
  row=self.auth.access(token)
  late=False; out=None
  if d.get('task_id') != task_id or d.get('task_type')!='system.echo' or d.get('status') not in ('completed','failed','rejected','cancelled','expired'): raise error('invalid_result')
  if not isinstance(d.get('stdout'),str) or not isinstance(d.get('stderr'),str) or len(d['stdout'].encode())>self.settings.max_stdout_bytes or len(d['stderr'].encode())>self.settings.max_stderr_bytes: raise error('payload_too_large')
  with self.store.transaction() as c:
   replay=self._dedup_replay(c,row,d,key,'POST','/worker/v1/tasks/{task_id}/result',task_id)
   if replay is not _NO_REPLAY: return replay
   self._assert_context(row,d); self._reap(c); delivery=c.execute("SELECT d.*,t.payload_hash,t.payload_json,t.trace_id,t.task_type,t.state AS task_state FROM worker_deliveries d JOIN worker_tasks t USING(task_id) WHERE d.delivery_id=? AND d.task_id=?",(d.get('delivery_id'),task_id)).fetchone()
   if not delivery: raise error('stale_delivery')
   if delivery['worker_id']!=row['worker_id'] or delivery['registration_id']!=row['registration_id']: raise error('worker_not_authorized')
   if delivery['task_type']!=d.get('task_type'): raise error('invalid_result')
   result_hash=canonical_json_hash(d); existing=c.execute("SELECT result_hash FROM worker_results WHERE task_id=? AND result_idempotency_key=?",(task_id,d.get('result_idempotency_key'))).fetchone()
   if existing:
    if existing['result_hash']!=result_hash: raise error('idempotency_conflict')
    out={'accepted':True,'duplicate':True,'task_state':delivery['task_state'],'server_time':self.now()}; self._dedup_store(c,row,d,key,'POST','/worker/v1/tasks/{task_id}/result',task_id,out,200); return out
   if delivery['state']=='expired':
    late=True
   elif delivery['state']!='acknowledged':
    raise error('state_conflict')
   else:
    if d.get('payload_hash')!=delivery['payload_hash'] or d.get('trace_id')!=delivery['trace_id']: raise error('invalid_result')
    message=json.loads(delivery['payload_json'])['message']
    if d['status']=='completed' and (d['stdout']!=message or d['stderr']!='' or d.get('exit_code')!=0): raise error('invalid_result')
    c.execute("INSERT INTO worker_results VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",(str(uuid.uuid4()),task_id,delivery['delivery_id'],d['result_idempotency_key'],result_hash,d['status'],d['stdout'],d['stderr'],d.get('exit_code'),d['started_at'],d['finished_at'],d['duration_ms'],self.now()))
    state='completed' if d['status']=='completed' else ('rejected' if d['status']=='rejected' else 'failed'); c.execute("UPDATE worker_deliveries SET state='completed',finished_at=? WHERE delivery_id=?",(self.now(),delivery['delivery_id'])); c.execute("UPDATE worker_tasks SET state=?,leased_until=NULL WHERE task_id=?",(state,task_id)); out={'accepted':True,'duplicate':False,'task_state':state,'server_time':self.now()}; self._dedup_store(c,row,d,key,'POST','/worker/v1/tasks/{task_id}/result',task_id,out,200); self._audit(c,'result_accepted',worker_id=row['worker_id'],task_id=task_id,delivery_id=delivery['delivery_id'],trace_id=delivery['trace_id'])
  if late: raise error('lease_expired')
  return out
 def task_state(self,task_id): return self.store.conn.execute("SELECT state FROM worker_tasks WHERE task_id=?",(task_id,)).fetchone()['state']
 def result_count(self,task_id): return self.store.conn.execute("SELECT count(*) FROM worker_results WHERE task_id=?",(task_id,)).fetchone()[0]
 def audit_text(self): return '\n'.join(str(tuple(r)) for r in self.store.conn.execute("SELECT * FROM worker_audit_log"))

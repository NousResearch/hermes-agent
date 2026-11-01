"""Self-contained persistent Graph-and-Loop state for the Hermes plugin."""
from __future__ import annotations
import json, threading, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
try:
    from hermes_constants import get_hermes_home
except Exception:
    def get_hermes_home() -> Path:
        return Path.home() / ".hermes"

def now(): return datetime.now(timezone.utc).isoformat()
def new_id(prefix): return f"{prefix}_{uuid.uuid4().hex[:12]}"

class Plane:
    def __init__(self):
        self.path=get_hermes_home()/"graph-loop-orchestrator"/"state.json"
        self.path.parent.mkdir(parents=True,exist_ok=True); self.lock=threading.RLock(); self.graphs={}; self.pings=[]; self.load()
    def load(self):
        try:
            d=json.loads(self.path.read_text(encoding="utf-8")); self.graphs=d.get("graphs",{}); self.pings=d.get("pings",[])[-500:]
        except (OSError,ValueError): self.graphs={}; self.pings=[]
    def save(self):
        tmp=self.path.with_suffix(".tmp"); tmp.write_text(json.dumps({"version":1,"updated_at":now(),"graphs":self.graphs,"pings":self.pings[-500:]},indent=2),encoding="utf-8"); tmp.replace(self.path)
    def get(self,gid):
        if gid not in self.graphs: raise KeyError(gid)
        return self.graphs[gid]
    def create(self,goal,dod,master="ceo",workers=None,nodes=None):
        if not str(goal).strip(): raise ValueError("goal is required")
        dod=[str(x).strip() for x in (dod or []) if str(x).strip()]
        if not dod: raise ValueError("definition_of_done must not be empty")
        gid=new_id("graph"); g={"id":gid,"goal":str(goal).strip(),"definition_of_done":dod,"master_agent":master or "ceo","workers":workers or [],"status":"planning","loop_iteration":0,"max_iterations":5,"created_at":now(),"updated_at":now(),"nodes":[],"verification":{"status":"pending","checks":[],"attempts":0},"events":[]}
        for raw in nodes or []: self.add_node(g,raw)
        self.event(g,"graph_created",{}); self.graphs[gid]=g; self.save(); return self.snapshot(gid)
    def add_node(self,g,raw):
        n={"id":raw.get("id") or new_id("node"),"title":str(raw.get("title") or raw.get("description") or "Worker task"),"description":str(raw.get("description") or raw.get("title") or ""),"kind":str(raw.get("kind") or "worker"),"assigned_to":str(raw.get("assigned_to") or raw.get("agent_id") or ""),"depends_on":list(raw.get("depends_on") or []),"status":str(raw.get("status") or "pending"),"result":raw.get("result"),"evidence":list(raw.get("evidence") or []),"attempts":0,"updated_at":now()}; g["nodes"].append(n); return n
    def snapshot(self,gid): return json.loads(json.dumps(self.get(gid)))
    def event(self,g,name,data): g["events"]=(g.get("events",[])+[{"at":now(),"event":name,"data":data}])[-200:]; g["updated_at"]=now()
    def runnable(self,gid):
        g=self.get(gid); done={n["id"] for n in g["nodes"] if n["status"]=="complete"}; return [json.loads(json.dumps(n)) for n in g["nodes"] if n["status"]=="pending" and set(n.get("depends_on",[]))<=done]
    def claim(self,gid,nid,agent=""):
        g=self.get(gid); n=next((x for x in g["nodes"] if x["id"]==nid),None)
        if not n: raise KeyError(nid)
        if n["status"] not in ("pending","retry"): raise ValueError(f"node is {n['status']}")
        n.update({"status":"working","assigned_to":agent or n["assigned_to"],"attempts":n["attempts"]+1,"updated_at":now()}); g["status"]="working"; self.event(g,"node_claimed",{"node_id":nid,"agent_id":agent}); self.save(); return n
    def complete(self,gid,nid,result,evidence=None,status="complete"):
        g=self.get(gid); n=next((x for x in g["nodes"] if x["id"]==nid),None)
        if not n: raise KeyError(nid)
        if status not in ("complete","error","retry"): raise ValueError("invalid status")
        n.update({"status":status,"result":result,"evidence":evidence or [],"updated_at":now()}); self.event(g,"node_finished",{"node_id":nid,"status":status})
        if g["nodes"] and all(x["status"]=="complete" for x in g["nodes"]): g["status"]="verifying"
        self.save(); return self.snapshot(gid)
    def verify(self,gid,checks):
        g=self.get(gid); g["verification"]["attempts"]+=1; out=[]
        for criterion in g["definition_of_done"]:
            c=next((x for x in checks if str(x.get("criterion",""))==criterion),None); out.append({"criterion":criterion,"passed":bool(c and c.get("passed") is True),"evidence":(c or {}).get("evidence","")})
        passed=bool(g["nodes"]) and all(x["status"]=="complete" for x in g["nodes"]) and all(x["passed"] for x in out)
        g["verification"].update({"status":"passed" if passed else "failed","checks":out,"verified_at":now()}); g["status"]="complete" if passed else "looping"
        if not passed: g["loop_iteration"]+=1; g["status"]="blocked" if g["loop_iteration"]>=g["max_iterations"] else g["status"]
        else: g["completed_at"]=now()
        self.event(g,"verification_finished",{"passed":passed}); self.save(); return self.snapshot(gid)
    def ping(self,sender,recipients,message,gid="",nid="",priority="normal"):
        if not recipients or not str(message).strip(): raise ValueError("recipients and message are required")
        p={"id":new_id("ping"),"sender":sender or "system","recipients":[str(x) for x in recipients],"message":str(message).strip(),"graph_id":gid,"node_id":nid,"priority":priority,"status":"queued","created_at":now(),"delivered_at":None}; self.pings.append(p); self.pings=self.pings[-500:]; self.save(); return p
    def status(self): return {"control_plane":"graph-and-loop","plugin":"graph-loop-orchestrator","version":1,"state_path":str(self.path),"graphs":len(self.graphs),"active_graphs":sum(g["status"] in ("planning","working","verifying","looping") for g in self.graphs.values()),"queued_pings":sum(p["status"]=="queued" for p in self.pings),"roles":{"master":"definition of done","worker":"node execution","verifier":"strict completion gate","router":"asynchronous pings"}}

plane=Plane()

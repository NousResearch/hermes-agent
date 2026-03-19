from pathlib import Path
from datetime import datetime
import json
from loguru import logger
import statistics

class SwarmMetrics:
    """
    Tracks the performance and health of the Swarm.
    Data is used to drive Self-Evolution decisions.
    """
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent.parent.parent
        self.metrics_file = self.project_root / "data" / "evolution" / "metrics.json"
        self._load_metrics()
        
    def _load_metrics(self):
        self.data = {
            "tasks": {}, # {task_type: {total: 0, success: 0, errors: []}}
            "tokens": {"in": 0, "out": 0, "total": 0},
            "system_health": [],
            "last_updated": datetime.now().isoformat()
        }
        try:
            if self.metrics_file.exists():
                content = self.metrics_file.read_text()
                if content:
                    self.data = json.loads(content)
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")

    def _save_metrics(self):
        try:
            self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
            self.data["last_updated"] = datetime.now().isoformat()
            self.metrics_file.write_text(json.dumps(self.data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")

    def log_task(self, task_type: str, success: bool, error: str = None, latency: float = 0.0):
        """
        Log the outcome of a swarm task (e.g., "code_generation", "chat_response").
        """
        if task_type not in self.data["tasks"]:
            self.data["tasks"][task_type] = {
                "total": 0, 
                "success": 0, 
                "latencies": [],
                "errors": {}
            }
            
        record = self.data["tasks"][task_type]
        record["total"] += 1
        if success:
            record["success"] += 1
        
        if latency > 0:
            record["latencies"].append(latency)
            # Keep only last 100 for rolling average
            if len(record["latencies"]) > 100:
                record["latencies"].pop(0)
                
        if error:
            safe_error = str(error)[:100] # Truncate for sanity
            record["errors"][safe_error] = record["errors"].get(safe_error, 0) + 1
            
        self._save_metrics()

    def get_success_rate(self, task_type: str) -> float:
        if task_type not in self.data["tasks"]:
            return 1.0 # Optimistic default
        record = self.data["tasks"][task_type]
        if record["total"] == 0:
            return 1.0
        return record["success"] / record["total"]
        
    def get_average_latency(self, task_type: str) -> float:
        if task_type not in self.data["tasks"]:
            return 0.0
        latencies = self.data["tasks"][task_type].get("latencies", [])
        if not latencies:
            return 0.0
        return statistics.mean(latencies)

    def log_token_usage(self, prompt_tokens: int, completion_tokens: int):
        """
        Log token usage for cost/throughput tracking.
        """
        if "tokens" not in self.data:
            self.data["tokens"] = {"in": 0, "out": 0, "total": 0}
            
        self.data["tokens"]["in"] += prompt_tokens
        self.data["tokens"]["out"] += completion_tokens
        self.data["tokens"]["total"] += (prompt_tokens + completion_tokens)
        
        # Save occasionally to avoid disk thrashing (every 1000 tokens or so?)
        # For now, just save.
        self._save_metrics()

# Global instance
swarm_metrics = SwarmMetrics()

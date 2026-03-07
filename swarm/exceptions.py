class SwarmError(Exception):
    pass


class TaskFailedError(SwarmError):
    def __init__(self, task_id: str, reason: str = ""):
        self.task_id = task_id
        self.reason = reason
        super().__init__(f"Task {task_id} failed: {reason}" if reason else f"Task {task_id} failed")


class WorkerUnavailableError(SwarmError):
    def __init__(self, worker_id: str = "", reason: str = ""):
        self.worker_id = worker_id
        self.reason = reason
        msg = f"Worker {worker_id} unavailable" if worker_id else "No workers available"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class BudgetExceededError(SwarmError):
    def __init__(self, spent: float = 0.0, limit: float = 0.0):
        self.spent = spent
        self.limit = limit
        super().__init__(f"Budget exceeded: ${spent:.4f} spent, ${limit:.4f} limit")


class DependencyFailedError(SwarmError):
    def __init__(self, task_id: str, dep_id: str):
        self.task_id = task_id
        self.dep_id = dep_id
        super().__init__(f"Dependency {dep_id} failed for task {task_id}")

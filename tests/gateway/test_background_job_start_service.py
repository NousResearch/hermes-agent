from gateway.background_job_start_service import start_background_job


class _Store:
    def __init__(self):
        self.created = None
        self.launcher_updates = []
        self.failures = []

    def create_job(self, **kwargs):
        self.created = dict(kwargs)
        return {"task_id": kwargs["task_id"], "status": "queued"}

    def update_job_launcher(self, task_id, metadata):
        self.launcher_updates.append((task_id, dict(metadata)))
        return {"task_id": task_id, "status": "queued", "launcher": dict(metadata)}

    def mark_job_failed(self, task_id, error):
        self.failures.append((task_id, error))
        return {"task_id": task_id, "status": "failed", "error": error}


def test_start_background_job_persists_and_launches_worker():
    store = _Store()

    task_id = start_background_job(
        store=store,
        launch_worker=lambda current_task_id: {
            "launcher_type": "subprocess",
            "launcher_pid": 4321,
            "task_id_seen": current_task_id,
        },
        prompt="继续排查 gateway 问题",
        source={"platform": "qq"},
        preloaded_skills=["frontend-design"],
        admin_user_ids=["179033731"],
        task_id_factory=lambda: "bg_test_001",
    )

    assert task_id == "bg_test_001"
    assert store.created["task_id"] == "bg_test_001"
    assert store.created["preloaded_skills"] == ["frontend-design"]
    assert store.launcher_updates == [
        (
            "bg_test_001",
            {
                "launcher_type": "subprocess",
                "launcher_pid": 4321,
                "task_id_seen": "bg_test_001",
            },
        )
    ]


def test_start_background_job_marks_failure_when_launcher_raises():
    store = _Store()

    task_id = start_background_job(
        store=store,
        launch_worker=lambda _task_id: (_ for _ in ()).throw(RuntimeError("boom")),
        prompt="继续排查 gateway 问题",
        source={"platform": "qq"},
        task_id_factory=lambda: "bg_test_002",
    )

    assert task_id == "bg_test_002"
    assert store.failures == [("bg_test_002", "boom")]

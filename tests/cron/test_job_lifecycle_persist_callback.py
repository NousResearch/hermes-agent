"""#122813: cron job register/release must notify lifecycle observers so the
gateway can persist ``active_agents`` at every count change, not only at inbound
turn boundaries (until now a cron job left gateway_state.json stale for its
whole run, and a count written mid-job stayed stuck after the job ended)."""
import cron.scheduler as sched


def _cleanup(job_id):
    sched.release_running_job(job_id)  # idempotent


def test_register_and_release_fire_callback():
    job_id = "persist-cb-test-1"
    calls = []

    def cb():
        calls.append(sched.get_running_job_ids())

    sched.register_job_lifecycle_callback(cb)
    try:
        assert sched.try_register_running_job(job_id) is True
        _cleanup(job_id)
    finally:
        sched.unregister_job_lifecycle_callback(cb)
    assert len(calls) == 2
    assert job_id in calls[0]        # registered: callback sees the job in-flight
    assert job_id not in calls[1]    # released: callback sees it gone


def test_callback_exception_swallowed():
    job_id = "persist-cb-test-2"

    def boom():
        raise RuntimeError("observer failed")

    after = []

    def ok():
        after.append(1)

    sched.register_job_lifecycle_callback(boom)
    sched.register_job_lifecycle_callback(ok)
    try:
        assert sched.try_register_running_job(job_id) is True
        _cleanup(job_id)
        # ok() still fired despite boom() raising
        assert len(after) == 2
    finally:
        sched.unregister_job_lifecycle_callback(boom)
        sched.unregister_job_lifecycle_callback(ok)


def test_unregister_stops_firing():
    job_id = "persist-cb-test-3"
    calls = []

    def cb():
        calls.append(1)

    sched.register_job_lifecycle_callback(cb)
    sched.unregister_job_lifecycle_callback(cb)
    assert sched.try_register_running_job(job_id) is True
    _cleanup(job_id)
    assert calls == []


def test_double_register_is_single_registration():
    job_id = "persist-cb-test-4"
    calls = []

    def cb():
        calls.append(1)

    sched.register_job_lifecycle_callback(cb)
    sched.register_job_lifecycle_callback(cb)
    try:
        assert sched.try_register_running_job(job_id) is True
        _cleanup(job_id)
    finally:
        sched.unregister_job_lifecycle_callback(cb)
    assert len(calls) == 2  # one register + one release, not duplicated

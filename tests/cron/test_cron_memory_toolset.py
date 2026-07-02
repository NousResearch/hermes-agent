from cron.scheduler import _cron_should_skip_memory


def test_cron_skips_memory_by_default():
    assert _cron_should_skip_memory(None) is True
    assert _cron_should_skip_memory([]) is True
    assert _cron_should_skip_memory(["terminal", "file"]) is True


def test_cron_initializes_memory_when_memory_toolset_enabled():
    assert _cron_should_skip_memory(["file", "memory", "skills"]) is False

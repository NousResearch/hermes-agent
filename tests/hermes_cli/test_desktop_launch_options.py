from hermes_cli.desktop_launch_options import normalize_desktop_disable_gpu


def test_normalize_desktop_disable_gpu_accepts_documented_values() -> None:
    assert normalize_desktop_disable_gpu(True) == "1"
    assert normalize_desktop_disable_gpu(False) == "0"
    assert normalize_desktop_disable_gpu(" YES ") == "1"
    assert normalize_desktop_disable_gpu("off") == "0"
    assert normalize_desktop_disable_gpu("auto") == "auto"
    assert normalize_desktop_disable_gpu("unexpected") == "auto"
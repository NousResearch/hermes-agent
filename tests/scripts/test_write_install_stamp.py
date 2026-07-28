from scripts.write_install_stamp import build_stamp


def test_build_stamp_keeps_provenance_separate_from_distribution():
    stamp = build_stamp(commit="a" * 40, source="ci", distribution="docker")

    assert stamp["source"] == "ci"
    assert stamp["distribution"] == "docker"
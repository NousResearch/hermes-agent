from conductor.capabilities import conductor_capabilities, reviewer_capabilities


def test_conductor_and_reviewer_have_no_product_write_or_repair_surface():
    assert conductor_capabilities() == frozenset({
        "dispatch",
        "observe",
        "verify_receipt",
    })
    assert reviewer_capabilities() == frozenset({"read", "review", "write_receipt"})
    forbidden = {"write_file", "patch", "terminal", "repair", "delegate"}
    assert not (conductor_capabilities() & forbidden)
    assert not (reviewer_capabilities() & forbidden)

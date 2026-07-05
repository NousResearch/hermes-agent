import pytest

from hermes_cli.workflows_expr import eval_condition, resolve_path


def test_resolve_path_nested_dict_and_list():
    data = {"node": {"review": {"output": {"items": [{"score": 0.9}]}}}}
    assert resolve_path(data, "$.node.review.output.items[0].score") == 0.9


def test_eval_condition_boolean_tree():
    data = {"review": {"verdict": "approved", "confidence": 0.92}}
    cond = {
        "op": "and",
        "args": [
            {"op": "eq", "left": {"path": "$.review.verdict"}, "right": "approved"},
            {"op": "gte", "left": {"path": "$.review.confidence"}, "right": 0.8},
        ],
    }
    assert eval_condition(cond, data) is True


def test_eval_condition_or_and_not():
    data = {"review": {"verdict": "rejected", "confidence": 0.92}}
    cond = {
        "op": "and",
        "args": [
            {
                "op": "or",
                "args": [
                    {"op": "eq", "left": {"path": "$.review.verdict"}, "right": "approved"},
                    {"op": "eq", "left": {"path": "$.review.verdict"}, "right": "rejected"},
                ],
            },
            {
                "op": "not",
                "arg": {"op": "lt", "left": {"path": "$.review.confidence"}, "right": 0.8},
            },
        ],
    }
    assert eval_condition(cond, data) is True


def test_eval_condition_exists_and_missing():
    data = {"review": {"verdict": "approved", "notes": None}}

    assert eval_condition({"op": "exists", "path": "$.review.verdict"}, data) is True
    assert eval_condition({"op": "exists", "path": "$.review.missing"}, data) is False
    assert eval_condition({"op": "missing", "path": "$.review.missing"}, data) is True
    assert eval_condition({"op": "missing", "path": "$.review.notes"}, data) is False


@pytest.mark.parametrize(
    ("op", "left", "right"),
    [
        ("ne", "approved", "rejected"),
        ("gt", 0.92, 0.8),
        ("lt", 0.7, 0.8),
        ("lte", 0.8, 0.8),
    ],
)
def test_eval_condition_comparison_ops(op, left, right):
    assert eval_condition({"op": op, "left": left, "right": right}, {}) is True


@pytest.mark.parametrize(
    ("op", "right"),
    [
        ("contains", "prove"),
        ("starts_with", "approve"),
        ("ends_with", "now"),
        ("regex", r"approved\s+now"),
    ],
)
def test_eval_condition_string_ops(op, right):
    assert eval_condition({"op": op, "left": "approved now", "right": right}, {}) is True


def test_eval_condition_missing_path_is_false_for_comparison():
    assert eval_condition({"op": "eq", "left": {"path": "$.missing"}, "right": 1}, {}) is False


def test_eval_condition_missing_path_is_false_for_string_op():
    assert (
        eval_condition({"op": "contains", "left": {"path": "$.missing"}, "right": 1}, {})
        is False
    )


@pytest.mark.parametrize(
    "cond",
    [
        {"op": "and"},
        {"op": "and", "args": []},
        {"op": "and", "args": "not conditions"},
        {"op": "or"},
        {"op": "or", "args": []},
        {"op": "not"},
        {"op": "not", "args": []},
        {
            "op": "not",
            "args": [
                {"op": "eq", "left": 1, "right": 1},
                {"op": "eq", "left": 2, "right": 2},
            ],
        },
        {"op": "exists"},
        {"op": "missing"},
        {"op": "eq", "left": 1},
        {"op": "contains", "left": "abc"},
    ],
)
def test_eval_condition_rejects_malformed_guards(cond):
    with pytest.raises(ValueError):
        eval_condition(cond, {})


@pytest.mark.parametrize(
    "cond",
    [
        {"op": "or", "args": [{"op": "eq", "left": 1, "right": 1}, {"op": "and"}]},
        {"op": "and", "args": [{"op": "eq", "left": 1, "right": 2}, {"op": "and"}]},
    ],
)
def test_eval_condition_rejects_malformed_short_circuited_branch(cond):
    with pytest.raises(ValueError):
        eval_condition(cond, {})


def test_eval_condition_invalid_regex_raises_value_error():
    with pytest.raises(ValueError):
        eval_condition({"op": "regex", "left": "approved", "right": "["}, {})


def test_eval_condition_rejects_unknown_op():
    try:
        eval_condition({"op": "exec", "code": "print('nope')"}, {})
    except ValueError as exc:
        assert "unsupported condition op" in str(exc)
    else:
        raise AssertionError("expected ValueError")

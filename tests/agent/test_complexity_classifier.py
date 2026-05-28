from agent.complexity_classifier import ComplexityClassifier, cascade_router_classify


def test_classifies_greetings_as_nano():
    assert ComplexityClassifier.classify("hello") == "nano"
    assert ComplexityClassifier.classify("спасибо") == "nano"


def test_short_commands_are_mini_not_nano():
    assert ComplexityClassifier.classify("write a small script") == "mini"
    assert ComplexityClassifier.classify("сделай файл") == "mini"


def test_debug_and_architecture_are_full_not_frontier():
    assert ComplexityClassifier.classify("debug this failing pytest case") == "full"
    assert ComplexityClassifier.classify("explain the architecture here") == "full"
    assert ComplexityClassifier.classify("дебаг почему тест падает") == "full"


def test_urgent_or_major_refactor_markers_are_frontier():
    assert ComplexityClassifier.classify("!urgent fix production now") == "frontier"
    assert ComplexityClassifier.classify("complete rewrite of the gateway") == "frontier"


def test_code_block_size_changes_complexity():
    small_code = "```python\nprint('hello')\n```"
    medium_code = "```python\n" + "\n".join(f"print({i})" for i in range(12)) + "\n```"
    large_code = "```python\n" + "\n".join(f"print({i})" for i in range(55)) + "\n```"

    assert ComplexityClassifier.classify(small_code) == "mini"
    assert ComplexityClassifier.classify(medium_code) == "full"
    assert ComplexityClassifier.classify(large_code) == "frontier"


def test_cascade_router_classify_delegates_to_classifier():
    assert cascade_router_classify("why does this happen?") == "full"

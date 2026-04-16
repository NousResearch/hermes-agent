from run_agent import AIAgent


def _agent():
    return AIAgent.__new__(AIAgent)


def test_renumber_repeated_dot_markers():
    agent = _agent()
    text = "1. a\n1. b\n1. c"
    assert agent._renumber_repeated_ordered_list_markers(text) == "1. a\n2. b\n3. c"


def test_renumber_repeated_paren_markers():
    agent = _agent()
    text = "1) a\n1) b"
    assert agent._renumber_repeated_ordered_list_markers(text) == "1) a\n2) b"


def test_renumber_repeated_cjk_markers():
    agent = _agent()
    text = "1、甲\n1、乙\n1、丙"
    assert agent._renumber_repeated_ordered_list_markers(text) == "1、甲\n2、乙\n3、丙"


def test_leave_non_repeated_or_mixed_numbering_unchanged():
    agent = _agent()
    text = "1. a\n2. b\n1. c"
    assert agent._renumber_repeated_ordered_list_markers(text) == text


def test_leave_single_item_unchanged():
    agent = _agent()
    text = "1. only"
    assert agent._renumber_repeated_ordered_list_markers(text) == text


def test_leave_blank_line_separated_lists_unchanged():
    agent = _agent()
    text = "1. a\n\n1. b"
    assert agent._renumber_repeated_ordered_list_markers(text) == text


def test_renumber_nested_list_by_indent_block_only():
    agent = _agent()
    text = "1. parent\n  1. child a\n  1. child b"
    assert agent._renumber_repeated_ordered_list_markers(text) == "1. parent\n  1. child a\n  2. child b"


def test_leave_fenced_code_block_unchanged():
    agent = _agent()
    text = "```\n1. code\n1. still code\n```"
    assert agent._renumber_repeated_ordered_list_markers(text) == text

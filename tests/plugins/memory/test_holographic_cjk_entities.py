"""Entity extraction must work for CJK facts, and stay unchanged for English.

`_extract_entities` used to rely entirely on ASCII capitalization, so a store
holding Korean/Chinese/Japanese facts kept `entities` and `fact_entities`
empty: `probe`/`related`/`reason` lost their structural signal and quietly
fell back to FTS5 keyword matching (issue #24416).

The CJK rules are gated on the text containing CJK, so the English behaviour
these tests pin down must not move, and un-bracketed candidates have to look
like names — an entities table full of ordinary prose words is its own bug
(issue #57900).
"""

import pytest

from plugins.memory.holographic.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path / "memory.db")
    yield s
    s.close()


def test_korean_fact_links_entities_end_to_end(store):
    fact_id = store.add_fact("팀은 회의록을 Notion에 정리한다")

    rows = store._conn.execute(
        "SELECT e.name FROM entities e "
        "JOIN fact_entities fe ON fe.entity_id = e.entity_id "
        "WHERE fe.fact_id = ?",
        (fact_id,),
    ).fetchall()

    assert [r["name"] for r in rows] == ["Notion"]


def test_particles_fall_off_latin_runs(store):
    # 는/에/를 attach with no space; the run stops at the first Hangul char.
    assert store._extract_entities("Hermes는 Notion에 GPT-5를 쓴다") == [
        "Hermes",
        "Notion",
        "GPT-5",
    ]


@pytest.mark.parametrize(
    "text, expected",
    [
        # Reproduction from issue #24416.
        (
            "用户公司日常用「白兔」/「白兔控股」，不要用工商执照名「成都抖咖」",
            ["白兔", "白兔控股", "成都抖咖"],
        ),
        ("『月報』は Slack で共有する", ["月報", "Slack"]),
        # The brackets are CJK punctuation, so rule 5 does not need other CJK.
        ("「Slack」", ["Slack"]),
    ],
)
def test_cjk_bracket_quotes(store, text, expected):
    assert store._extract_entities(text) == expected


def test_multiword_name_stays_one_entity_across_scripts(store):
    # John and Doe stay together; Q3 sits inside English prose, away from any
    # CJK character, so the run rule leaves it alone.
    assert store._extract_entities("The Q3 report은 John Doe가 작성") == ["John Doe"]


@pytest.mark.parametrize(
    "text, expected",
    [
        # One CJK character is enough to open the rule on the whole sentence,
        # so an English clause must not turn into entities (#57900). The clause
        # can sit on either side of the CJK.
        ("Today we discussed 中文", []),
        ("결론: 中文 We should ship tomorrow", []),
        ("팀 회의 Today was long", []),
        ("Meeting notes 会議 Follow up needed", []),
        ("이건 The 방법 이다", []),
        # A lowercase word must not swallow the name behind it: the run has to
        # match at Notion, not at "in".
        ("Please save this in Notion 中文", ["Notion"]),
        ("회의는 Zoom 에서 한다", ["Zoom"]),
    ],
)
def test_english_prose_in_cjk_facts_is_not_an_entity(store, text, expected):
    assert store._extract_entities(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        # An internal capital is part of the name, not the start of one.
        ("iPhone은 비싸다", ["iPhone"]),
        ("macOS에서 실행", ["macOS"]),
        ("팀은 eBay를 쓴다", ["eBay"]),
    ],
)
def test_internal_capitals_stay_whole(store, text, expected):
    assert store._extract_entities(text) == expected


def test_names_in_a_list_all_survive(store):
    # A name is followed by CJK, by another name, or by nothing.
    assert store._extract_entities("Notion, Slack 그리고 Jira") == [
        "Notion",
        "Slack",
        "Jira",
    ]
    assert store._extract_entities("Slack\n\n중요") == ["Slack"]


def test_lowercase_prose_is_not_an_entity(store):
    assert store._extract_entities("그 파일을 download 했다") == []
    assert store._extract_entities("코드는 don't use it 이다") == []
    assert store._extract_entities("이건 the best 방법 a b") == []


def test_urls_and_emails_are_not_split_into_entities(store):
    assert store._extract_entities("문서는 https://www.notion.so/team/abc-123 에 있다") == []
    assert store._extract_entities("이메일은 foo.bar@example.com 이다") == []


@pytest.mark.parametrize(
    "text, expected",
    [
        ("John Doe joined the team", ["John Doe"]),
        ('The tool is "Python" here', ["Python"]),
        ("Guido aka BDFL", ["Guido", "BDFL"]),
        ("Active users grew and the report is fine", []),
        ("download the file from https://example.com/a_b", []),
    ],
)
def test_english_extraction_is_unchanged(store, text, expected):
    assert store._extract_entities(text) == expected

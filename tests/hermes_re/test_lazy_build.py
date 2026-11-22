"""Lazy build: `compile()` checks and translates, the RE2 program is built on first use.

What must hold: every construct the facade refuses still raises at `compile()`; nothing but the
memory budget is left to first use; `build` / `build_all` force the
build; the registry behind `build_all` is weak; threads never see a half-built Pattern.
"""
import gc
import re as std
import threading
import weakref

import hermes_re as hre
from tests.hermes_re._support import outcome, view


def test_compile_does_not_build_and_first_use_does():
    hre.purge()
    compiled = hre.compile(r"(?P<key>\w+)=(\d+)$", hre.I)
    assert compiled._built is False
    assert compiled.pattern == r"(?P<key>\w+)=(\d+)$" and int(compiled.flags) == 34
    assert compiled.groups == 2 and compiled.groupindex == {"key": 1}  # known from the text alone
    assert repr(compiled).startswith("re.compile(") and hash(compiled) == hash(hre.compile(compiled.pattern, hre.I))
    assert compiled._built is False  # none of the above needed the program
    assert compiled.search("a=1\n").span() == (0, 3)
    assert compiled._built is True


def test_every_matching_operation_builds():
    calls = {
        "search": lambda p: p.search("ab"), "match": lambda p: p.match("ab"), "fullmatch": lambda p: p.fullmatch("ab"),
        "sub": lambda p: p.sub("", "ab"), "subn": lambda p: p.subn("", "ab"), "split": lambda p: p.split("ab"),
        "findall": lambda p: p.findall("ab"), "finditer": lambda p: p.finditer("ab"),
    }
    for number, (name, call) in enumerate(calls.items()):
        compiled = hre.compile(f"a{{1,{number + 2}}}b")  # a pattern no other test has built
        assert compiled._built is False, name
        call(compiled)
        assert compiled._built is True, name
    hre.purge()
    assert hre.search(r"lazy-module-level-\d", "x lazy-module-level-7").group() == "lazy-module-level-7"


def test_groups_and_groupindex_come_from_the_text_and_equal_stdlib():
    sources = [r"(a)(?:b)(?P<n>c)", r"(?P<a>x)|(?P<b>y)(z)", r"((a)|(?P<in>b))+", r"\((a)\)[(](b)", r"(?i:(a))(?-i:(b))",
               r"(?<!x)(?P<w>\w+)(?=\s|$)", r"(a)$", r"(?P<q>a)(?!\w)", r"^(?P<k>\w+)\s*=\s*(?P<v>.*)$", r"(?#(c)(a)",
               r"\b(?P<word>\w+)\b", r"(?:(?P<x>a)|b)(?=(c))"]
    for source in sources:
        mine, ref = hre.compile(source), std.compile(source)
        assert mine._built is False or mine.groups == ref.groups
        assert (mine.groups, mine.groupindex) == (ref.groups, dict(ref.groupindex)), source
        assert list(mine.groupindex) == list(ref.groupindex), source  # same order
        hre.build(mine)  # the build checks RE2's own group count against the text's
        assert mine.groups == ref.groups
    assert hre.compile(rb"(?P<n>a)(b)").groupindex == {"n": 1}


def test_what_the_facade_refuses_raises_at_compile_without_any_build():
    refused = [r"a(?=b)c", r"(?<=ab)c", r"x(?!ab)", r"(?<!a)(?<!b)x", r"(a)\1", r"(?P<n>a)(?P=n)", r"a++", r"a*+", r"(?>a)",
               r"(a)?(?(1)b|c)", r"\N{EM DASH}", r"(?a)\w", r"(?L)a", r"(?x) a", r"x$y", r"(x$)+", "\ud800", "a\\",
               r"(", r")", r"[a", r"[z-a]", r"*a", r"a**", r"a{2}{3}", r"a{3,2}", r"a{1001}", r"(?:a{40}){30}",
               r"(?:(?:a{10}){10}){11}", r"\q", r"\x4g", r"\u12", r"\8", r"(?P<1a>x)", r"(?P<a>x)(?P<a>y)", r"(?P<n", r"(?i",
               r"^*", r"\b+", r"\Qa\E", r"\C", r"\p{Xx}", r"\p{Klingon}", r"(?#unterminated"]
    hre.purge()
    gc.collect()
    for source in refused:
        try:
            hre.compile(source)
        except hre.error as exc:
            assert exc.pattern == source
        else:
            raise AssertionError(f"{source!r} was accepted")
    assert hre.build_all() == []  # nothing refused was registered


def test_only_the_budget_is_left_to_first_use():
    hre.purge()
    gc.collect()
    for source, phrase in ((r"\w{1000}" * 4, "too large"),):
        compiled = hre.compile(source)  # accepted: the program size cannot be told from the text
        for call in (lambda: compiled.search("x"), lambda: compiled.sub("", "x"), lambda: hre.build(compiled),
                     lambda: hre.search(source, "x")):
            try:
                call()
            except hre.error as exc:
                assert phrase in exc.msg and exc.pattern == source
            else:
                raise AssertionError(f"{source[:20]!r} built")
        assert compiled._built is False
        assert [(p.pattern, phrase in e.msg) for p, e in hre.build_all()] == [(source, True)]
        hre.purge()
        compiled = None
        gc.collect()
    assert hre.build_all() == []


def test_build_forces_the_program_and_returns_the_cached_pattern():
    hre.purge()
    built = hre.build(r"forced-\d+", hre.I)
    assert built is hre.compile(r"forced-\d+", hre.I) and built._built is True
    assert hre.build(built) is built
    assert hre.build(std.compile(r"std-\d+"))._built is True  # a compiled stdlib pattern is converted, then built
    assert outcome(lambda: hre.build(built, hre.I)) == "raises ValueError"


def test_build_all_builds_every_live_pattern_and_reports_failures():
    hre.purge()
    gc.collect()
    held = [hre.compile(source) for source in (r"held-a\d", r"(?<!x)held-b$", r"\bheld-c\b")]
    hre.search(r"cached-only-\w+", "cached-only-x")
    fresh = hre.compile(r"cached-unused-\w+")
    assert not any(p._built for p in held) and fresh._built is False
    assert hre.build_all() == []
    assert all(p._built for p in held) and fresh._built is True
    assert isinstance(held[2]._wide, str)          # the Unicode `\b` form is still text ...
    assert hre.build_all(wide=True) == []
    assert isinstance(held[2]._wide, tuple)        # ... until asked for
    bad = hre.compile(r"[\w.-]{1000}" * 4)
    failures = hre.build_all()
    assert [(p is bad, type(e) is hre.error) for p, e in failures] == [(True, True)]
    hre.purge()


def test_the_registry_is_weak_and_does_not_defeat_cache_eviction():
    from hermes_re import _pattern
    hre.purge()
    gc.collect()
    before = len(_pattern.live())
    compiled = hre.compile(r"weakly-registered-\d")
    ref = weakref.ref(compiled)
    assert compiled in _pattern.live() and len(_pattern.live()) == before + 1
    del compiled
    gc.collect()
    assert ref() is not None  # the compile cache still holds it
    hre.purge()               # ... evicted (purge here; LRU eviction drops the same single reference)
    gc.collect()
    assert ref() is None and len(_pattern.live()) == before


def test_threads_sharing_an_unbuilt_pattern_all_get_complete_results():
    from hermes_re import _form
    real_build, builds, gate = _form.Form.build, [], {}

    def slow_build(self):
        builds.append(threading.get_ident())
        if len(builds) > gate["before"]:  # a second thread found the pattern unbuilt as well
            gate["two"].set()
        gate["two"].wait(timeout=10)  # the first builder goes on only then (the deadline never passes)
        return real_build(self)

    cases = [(r"(\w+)-(\d+)", "ab-12 cd-34"), (r"(?<![\w.])(\w+)(?=\s|$)", "ab cd.ef gh"), (r"\b(\w+)\b", "na\u00efve caf\u00e9"),
             (r"(?i)(k+)$", "x\u212aK\n")]
    _form.Form.build = slow_build
    try:
        for round_number in range(3):
            for source, text in cases:
                hre.purge()
                compiled = hre.compile(source)
                assert compiled._built is False
                expected = [view(m) for m in std.compile(source).finditer(text)]
                barrier, results = threading.Barrier(12), []
                gate["before"], gate["two"] = len(builds) + 1, threading.Event()

                def work(compiled=compiled, text=text, barrier=barrier, results=results):
                    barrier.wait()
                    try:
                        results.append([view(m) for m in compiled.finditer(text)] + [compiled.groups])
                    except Exception as exc:  # a half-built pattern would show up here
                        results.append(repr(exc))

                threads = [threading.Thread(target=work) for _ in range(12)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
                assert results == [expected + [compiled.groups]] * 12, (source, round_number, results[:2])
                assert compiled._built is True and gate["two"].is_set()
    finally:
        _form.Form.build = real_build
    assert len(builds) >= len(cases) * 3  # built at least once per pattern; a concurrent second build is harmless

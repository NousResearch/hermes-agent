"""Caches, the error class, unsupported flags, stdlib-pattern conversion, and a missing engine."""
import re as std

import hermes_re as hre
from tests.hermes_re._support import fresh_interpreter, outcome


def test_compile_returns_the_cached_object():
    assert hre.compile(r"a+b", hre.I) is hre.compile(r"a+b", 2)
    assert hre.compile(r"a+b") is not hre.compile(r"a+b", hre.I)
    assert hre.compile("a").pattern == "a" and hre.compile(b"a").pattern == b"a"  # keyed by pattern type too
    compiled = hre.compile("x")
    assert hre.compile(compiled) is compiled


def test_purge_clears_the_three_caches():
    from hermes_re import _budget, _subject, _template
    big = "\u00e9" * 2000
    hre.search(r"q", big)
    hre.sub(r"(q)", r"<\1>", "q")
    assert _budget.held()[0] > 0
    assert _subject._last_subject[0] is big
    assert _template.parse.cache_info().currsize > 0
    hre.purge()
    assert _budget.held() == (0, 0)
    assert _subject._last_subject[0] is None
    assert _template.parse.cache_info().currsize == 0


def test_subject_cache_holds_one_mid_sized_text_with_its_offset_index():
    from hermes_re import _subject
    hre.purge()
    hre.search("a", "short \u00e9")
    assert _subject._last_subject[0] is None  # under 1,024 characters: not cached
    big = "\u00e9" * 5000 + "a"
    assert hre.search("a", big).span() == (5000, 5001)
    assert _subject._last_subject[0] is big and _subject._last_subject[3] is not None
    assert hre.compile("a").search(big, 4000).span() == (5000, 5001)


def test_error_is_a_subclass_of_stdlib_error_with_msg_and_pattern():
    assert issubclass(hre.error, std.error) and hre.PatternError is hre.error
    try:
        hre.compile(r"a(?=b)c")
    except std.error as exc:  # an existing `except re.error:` handler catches it
        assert type(exc) is hre.error
        assert exc.pattern == r"a(?=b)c" and "lookahead" in exc.msg
        assert exc.pos is None and exc.lineno is None and exc.colno is None
    else:
        raise AssertionError("no error raised")


def test_compile_is_eager():
    for bad in (r"(a)\1", r"a++", r"(", r"[a", r"*a", r"a{1001}", r"x$y", "\ud800", r"a(?=b)c", r"(?P<n>a)(?P=n)",
                r"(?>a)", r"(a)?(?(1)b)", r"\N{DASH}", r"(?a)\w", r"[z-a]", r"a**", r"\q", "a\\", r"(?:a{40}){30}"):
        assert outcome(lambda: hre.compile(bad)) == "raises error", bad
        assert outcome(lambda: hre.search(bad, "a")) == "raises error", bad


def test_unsupported_flags_raise_value_error_at_compile():
    for flags in (hre.X, hre.VERBOSE, hre.L, hre.LOCALE, hre.DEBUG, hre.I | hre.X, 64):
        try:
            hre.compile("a", flags)
        except ValueError as exc:
            assert "RE2 cannot honour" in str(exc)
        else:
            raise AssertionError(f"flags {flags!r} accepted")
    assert outcome(lambda: hre.compile("a", hre.A | hre.U)) == "raises ValueError"
    assert outcome(lambda: hre.compile(b"a", hre.U)) == "raises ValueError"


def test_a_compiled_stdlib_pattern_is_converted():
    ref = std.compile(r"(?P<w>\w+)$", std.IGNORECASE | std.MULTILINE)
    mine = hre.compile(ref)
    assert type(mine) is hre.Pattern and mine.pattern == ref.pattern and int(mine.flags) == int(ref.flags)
    assert mine is hre.compile(ref.pattern, hre.I | hre.M)
    assert hre.search(ref, "ab CD\n").span() == ref.search("ab CD\n").span()
    assert hre.sub(std.compile(r"\d"), "#", "a1") == "a#"
    assert hre.findall(std.compile(rb"\d+"), b"a1 22") == [b"1", b"22"]
    assert hre.split(std.compile(","), "a,b") == ["a", "b"]
    assert outcome(lambda: hre.search(std.compile(r"(a)\1"), "aa")) == "raises error"  # a conversion, not a fallback
    assert outcome(lambda: hre.compile(ref, hre.I)) == "raises ValueError"


def test_importing_the_facade_does_not_import_the_engine():
    out = fresh_interpreter("import sys, hermes_re; print('re2' in sys.modules, int(hermes_re.I | hermes_re.M)); "
                 "p = hermes_re.compile(r'(\\w+)$', 2); print('re2' in sys.modules, p.groups); "
                 "p.search('a'); print('re2' in sys.modules)")
    assert out == ["False", "10", "False", "1", "True"]  # compile() needs no engine; the first match loads it


def test_engine_missing_compile_succeeds_and_matching_raises_module_not_found():
    code = r'''
import sys
sys.modules["re2"] = None          # `import re2` now fails in this process
import hermes_re as re
WORD = re.compile(r"(?P<k>\w+)=(\d+)$", re.I)    # a module-level compile must not raise
print(type(WORD).__name__, WORD.pattern == r"(?P<k>\w+)=(\d+)$", int(WORD.flags))
try:
    re.compile("a(?=b)c")                       # the check needs no engine: an invalid pattern still raises
except re.error:
    print("error")
print(WORD.groups, sorted(WORD.groupindex.items()))   # known from the pattern text
calls = {"search": lambda: WORD.search("a=1"), "match": lambda: WORD.match("a=1"), "fullmatch": lambda: WORD.fullmatch("a=1"),
         "sub": lambda: WORD.sub("", "a=1"), "subn": lambda: WORD.subn("", "a=1"), "split": lambda: WORD.split("a=1"),
         "findall": lambda: WORD.findall("a=1"), "finditer": lambda: WORD.finditer("a=1"),
         "re.search": lambda: re.search("a", "a"), "re.sub": lambda: re.sub("a", "", "a"), "re.split": lambda: re.split("a", "a"),
         "build": lambda: re.build(WORD)}
for name, call in calls.items():
    try:
        call()
        print(name, "no-error")
    except ImportError as exc:
        print(name, type(exc).__name__, exc.name, "google-re2" in str(exc))
try:
    re.compile("a", re.X)
except ValueError:
    print("flags-still-checked")
'''
    out = fresh_interpreter(code)
    assert out[:8] == ["Pattern", "True", "34", "error", "2", "[('k',", "1)]", "search"], out[:8]
    names = ["search", "match", "fullmatch", "sub", "subn", "split", "findall", "finditer", "re.search", "re.sub", "re.split",
             "build"]
    assert out[7:-1] == [word for name in names for word in (name, "ModuleNotFoundError", "re2", "True")], out
    assert out[-1] == "flags-still-checked"


def test_engine_missing_policy_is_one_function():
    from hermes_re import _engine
    raised = _engine.engine_missing(ImportError("x"))  # the exception the first matching operation raises
    assert type(raised) is ModuleNotFoundError and raised.name == "re2" and "google-re2" in str(raised)
    assert "never falls back to stdlib re" in str(raised)


def test_engine_missing_has_the_shape_shipped_updaters_tolerate():
    """What the import-health probe of releases without google-re2 does with an exception: a
    ``ModuleNotFoundError`` whose top-level ``name`` is not first-party is "dependencies not installed
    yet"; every other ``ImportError`` is a broken checkout. The engine-missing error must be the former
    for both ways the import can fail: package absent, and package present but its extension unloadable."""
    code = r'''
import importlib.abc, sys
class Broken(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("re2", "_re2"):
            raise ImportError("libre2.so.11: cannot open shared object file", name=None)
if sys.argv[1] == "broken":
    sys.meta_path.insert(0, Broken())
else:
    sys.modules["re2"] = None
import hermes_re
for call in (lambda: hermes_re.search("a", "a"), lambda: hermes_re.compile("a").match("a")):   # the second call: no cached verdict
    try:
        call()
        print("no-error")
    except ModuleNotFoundError as exc:
        missing = (exc.name or "").split(".")[0]
        tolerated = bool(missing) and not missing.startswith("hermes_")   # "hermes_re" would read as first-party
        print(type(exc).__name__, exc.name, tolerated, type(exc.__cause__).__name__)
    except ImportError as exc:
        print("plain-ImportError", exc.name)
'''
    for how, cause in (("absent", "ModuleNotFoundError"), ("broken", "ImportError")):
        out = fresh_interpreter(code.replace("sys.argv[1]", repr(how)))
        assert out == ["ModuleNotFoundError", "re2", "True", cause] * 2, (how, out)

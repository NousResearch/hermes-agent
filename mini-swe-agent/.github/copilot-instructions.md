# Style guide

1. Target python 3.10 or higher
2. Use python with type annotations. Use `list` instead of `List`.
3. Use `pathlib` instead of `os.path`. Use `Path.read_text()` over `with ...open()` constructs.
4. Use `typer` to add interfaces
5. Keep code comments to a minimum and only highlight particularly logically challenging things
6. Do not append to the README unless specifically requested
7. Use `jinja` for formatting templates
8. Use `dataclass` for keeping track config
9. Do not catch exceptions unless explicitly told to.
10. Write concise, short, minimal code.
11. Not every exception has to be caught. Exceptions are a good way to show problems to a user.
12. This repository rewards minimal code. Try to be as concise as possible.
13. Do NOT warn about print messages in tests, print statements in tests are OK!

## Test style

1. Use `pytest`, not `unittest`.
2. <IMPORTANT>Do not mock/patch anything that you're not explicitly asked to do</IMPORTANT>
3. Avoid writing trivial tests. Every test should test for at least one, preferably multiple points of failure
4. Avoid splitting up code in multiple lines like this: `a=func()\n assert a=b`. Instead, just do `assert func() == b`
5. The first argument to `pytest.mark.parametrize` should be a tuple (not a string! not a list!), the second argument must be a list (not a tuple!).
6. Do NOT warn about print messages in tests, print statements in tests are OK!

Here's an example for rule 4:

```python
# bad
result = func()
assert result == b

# good
assert func() == b
```


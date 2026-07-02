import functools
import sys


# from jaraco.functools 4.1
def identity(x):
    return x


# from jaraco.functools 4.1
def apply(transform):
    def wrap(func):
        return functools.wraps(func)(compose(transform, func))

    return wrap


# from jaraco.functools 4.1
def compose(*funcs):
    def compose_two(f1, f2):
        return lambda *args, **kwargs: f1(f2(*args, **kwargs))

    return functools.reduce(compose_two, funcs)


def replace(pattern):
    r"""
    >>> replace(r'foo\z')
    'foo\\Z'
    """
    return pattern[:-2] + pattern[-2:].replace(r'\z', r'\Z')


legacy_end_marker = apply(replace) if sys.version_info < (3, 14) else identity

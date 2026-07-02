"""Imported from the recipes section of the itertools documentation.

All functions taken from the recipes section of the itertools library docs
[1]_.
Some backward-compatible usability improvements have been made.

.. [1] http://docs.python.org/library/itertools.html#recipes

"""

import random

from bisect import bisect_left, insort
from collections import deque
from contextlib import suppress
from functools import lru_cache, partial, reduce
from heapq import heappush, heappushpop
from itertools import (
    accumulate,
    chain,
    combinations,
    compress,
    count,
    cycle,
    groupby,
    islice,
    product,
    repeat,
    starmap,
    takewhile,
    tee,
    zip_longest,
)
from math import prod, comb, isqrt, gcd
from operator import mul, not_, itemgetter, getitem, index
from random import randrange, sample, choice
from sys import hexversion

__all__ = [
    'all_equal',
    'batched',
    'before_and_after',
    'consume',
    'convolve',
    'dotproduct',
    'first_true',
    'factor',
    'flatten',
    'grouper',
    'is_prime',
    'iter_except',
    'iter_index',
    'loops',
    'matmul',
    'multinomial',
    'ncycles',
    'nth',
    'nth_combination',
    'padnone',
    'pad_none',
    'pairwise',
    'partition',
    'polynomial_eval',
    'polynomial_from_roots',
    'polynomial_derivative',
    'powerset',
    'prepend',
    'quantify',
    'reshape',
    'random_combination_with_replacement',
    'random_combination',
    'random_permutation',
    'random_product',
    'repeatfunc',
    'roundrobin',
    'running_median',
    'sieve',
    'sliding_window',
    'subslices',
    'sum_of_squares',
    'tabulate',
    'tail',
    'take',
    'totient',
    'transpose',
    'triplewise',
    'unique',
    'unique_everseen',
    'unique_justseen',
]

_marker = object()


# zip with strict is available for Python 3.10+
try:
    zip(strict=True)
except TypeError:  # pragma: no cover
    _zip_strict = zip
else:  # pragma: no cover
    _zip_strict = partial(zip, strict=True)


# math.sumprod is available for Python 3.12+
try:
    from math import sumprod as _sumprod
except ImportError:  # pragma: no cover
    _sumprod = lambda x, y: dotproduct(x, y)


# heapq max-heap functions are available for Python 3.14+
try:
    from heapq import heappush_max, heappushpop_max
except ImportError:  # pragma: no cover
    _max_heap_available = False
else:  # pragma: no cover
    _max_heap_available = True


def take(n, iterable):
    """Return first *n* items of the *iterable* as a list.

        >>> take(3, range(10))
        [0, 1, 2]

    If there are fewer than *n* items in the iterable, all of them are
    returned.

        >>> take(10, range(3))
        [0, 1, 2]

    """
    return list(islice(iterable, n))


def tabulate(function, start=0):
    """Return an iterator over the results of ``func(start)``,
    ``func(start + 1)``, ``func(start + 2)``...

    *func* should be a function that accepts one integer argument.

    If *start* is not specified it defaults to 0. It will be incremented each
    time the iterator is advanced.

        >>> square = lambda x: x ** 2
        >>> iterator = tabulate(square, -3)
        >>> take(4, iterator)
        [9, 4, 1, 0]

    """
    return map(function, count(start))


def tail(n, iterable):
    """Return an iterator over the last *n* items of *iterable*.

    >>> t = tail(3, 'ABCDEFG')
    >>> list(t)
    ['E', 'F', 'G']

    """
    try:
        size = len(iterable)
    except TypeError:
        return iter(deque(iterable, maxlen=n))
    else:
        return islice(iterable, max(0, size - n), None)


def consume(iterator, n=None):
    """Advance *iterable* by *n* steps. If *n* is ``None``, consume it
    entirely.

    Efficiently exhausts an iterator without returning values. Defaults to
    consuming the whole iterator, but an optional second argument may be
    provided to limit consumption.

        >>> i = (x for x in range(10))
        >>> next(i)
        0
        >>> consume(i, 3)
        >>> next(i)
        4
        >>> consume(i)
        >>> next(i)
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
        StopIteration

    If the iterator has fewer items remaining than the provided limit, the
    whole iterator will be consumed.

        >>> i = (x for x in range(3))
        >>> consume(i, 5)
        >>> next(i)
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
        StopIteration

    """
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)


def nth(iterable, n, default=None):
    """Returns the nth item or a default value.

    >>> l = range(10)
    >>> nth(l, 3)
    3
    >>> nth(l, 20, "zebra")
    'zebra'

    """
    return next(islice(iterable, n, None), default)


def all_equal(iterable, key=None):
    """
    Returns ``True`` if all the elements are equal to each other.

        >>> all_equal('aaaa')
        True
        >>> all_equal('aaab')
        False

    A function that accepts a single argument and returns a transformed version
    of each input item can be specified with *key*:

        >>> all_equal('AaaA', key=str.casefold)
        True
        >>> all_equal([1, 2, 3], key=lambda x: x < 10)
        True

    """
    iterator = groupby(iterable, key)
    for first in iterator:
        for second in iterator:
            return False
        return True
    return True


def quantify(iterable, pred=bool):
    """Return the how many times the predicate is true.

    >>> quantify([True, False, True])
    2

    """
    return sum(map(pred, iterable))


def pad_none(iterable):
    """Returns the sequence of elements and then returns ``None`` indefinitely.

        >>> take(5, pad_none(range(3)))
        [0, 1, 2, None, None]

    Useful for emulating the behavior of the built-in :func:`map` function.

    See also :func:`padded`.

    """
    return chain(iterable, repeat(None))


padnone = pad_none


def ncycles(iterable, n):
    """Returns the sequence elements *n* times

    >>> list(ncycles(["a", "b"], 3))
    ['a', 'b', 'a', 'b', 'a', 'b']

    """
    return chain.from_iterable(repeat(tuple(iterable), n))


def dotproduct(vec1, vec2):
    """Returns the dot product of the two iterables.

    >>> dotproduct([10, 15, 12], [0.65, 0.80, 1.25])
    33.5
    >>> 10 * 0.65 + 15 * 0.80 + 12 * 1.25
    33.5

    In Python 3.12 and later, use ``math.sumprod()`` instead.
    """
    return sum(map(mul, vec1, vec2))


def flatten(listOfLists):
    """Return an iterator flattening one level of nesting in a list of lists.

        >>> list(flatten([[0, 1], [2, 3]]))
        [0, 1, 2, 3]

    See also :func:`collapse`, which can flatten multiple levels of nesting.

    """
    return chain.from_iterable(listOfLists)


def repeatfunc(func, times=None, *args):
    """Call *func* with *args* repeatedly, returning an iterable over the
    results.

    If *times* is specified, the iterable will terminate after that many
    repetitions:

        >>> from operator import add
        >>> times = 4
        >>> args = 3, 5
        >>> list(repeatfunc(add, times, *args))
        [8, 8, 8, 8]

    If *times* is ``None`` the iterable will not terminate:

        >>> from random import randrange
        >>> times = None
        >>> args = 1, 11
        >>> take(6, repeatfunc(randrange, times, *args))  # doctest:+SKIP
        [2, 4, 8, 1, 8, 4]

    """
    if times is None:
        return starmap(func, repeat(args))
    return starmap(func, repeat(args, times))


def _pairwise(iterable):
    """Returns an iterator of paired items, overlapping, from the original

    >>> take(4, pairwise(count()))
    [(0, 1), (1, 2), (2, 3), (3, 4)]

    On Python 3.10 and above, this is an alias for :func:`itertools.pairwise`.

    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


try:
    from itertools import pairwise as itertools_pairwise
except ImportError:  # pragma: no cover
    pairwise = _pairwise
else:  # pragma: no cover

    def pairwise(iterable):
        return itertools_pairwise(iterable)

    pairwise.__doc__ = _pairwise.__doc__


class UnequalIterablesError(ValueError):
    def __init__(self, details=None):
        msg = 'Iterables have different lengths'
        if details is not None:
            msg += (': index 0 has length {}; index {} has length {}').format(
                *details
            )

        super().__init__(msg)


def _zip_equal_generator(iterables):
    for combo in zip_longest(*iterables, fillvalue=_marker):
        for val in combo:
            if val is _marker:
                raise UnequalIterablesError()
        yield combo


def _zip_equal(*iterables):
    # Check whether the iterables are all the same size.
    try:
        first_size = len(iterables[0])
        for i, it in enumerate(iterables[1:], 1):
            size = len(it)
            if size != first_size:
                raise UnequalIterablesError(details=(first_size, i, size))
        # All sizes are equal, we can use the built-in zip.
        return zip(*iterables)
    # If any one of the iterables didn't have a length, start reading
    # them until one runs out.
    except TypeError:
        return _zip_equal_generator(iterables)


def grouper(iterable, n, incomplete='fill', fillvalue=None):
    """Group elements from *iterable* into fixed-length groups of length *n*.

    >>> list(grouper('ABCDEF', 3))
    [('A', 'B', 'C'), ('D', 'E', 'F')]

    The keyword arguments *incomplete* and *fillvalue* control what happens for
    iterables whose length is not a multiple of *n*.

    When *incomplete* is `'fill'`, the last group will contain instances of
    *fillvalue*.

    >>> list(grouper('ABCDEFG', 3, incomplete='fill', fillvalue='x'))
    [('A', 'B', 'C'), ('D', 'E', 'F'), ('G', 'x', 'x')]

    When *incomplete* is `'ignore'`, the last group will not be emitted.

    >>> list(grouper('ABCDEFG', 3, incomplete='ignore', fillvalue='x'))
    [('A', 'B', 'C'), ('D', 'E', 'F')]

    When *incomplete* is `'strict'`, a subclass of `ValueError` will be raised.

    >>> iterator = grouper('ABCDEFG', 3, incomplete='strict')
    >>> list(iterator)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    UnequalIterablesError

    """
    iterators = [iter(iterable)] * n
    if incomplete == 'fill':
        return zip_longest(*iterators, fillvalue=fillvalue)
    if incomplete == 'strict':
        return _zip_equal(*iterators)
    if incomplete == 'ignore':
        return zip(*iterators)
    else:
        raise ValueError('Expected fill, strict, or ignore')


def roundrobin(*iterables):
    """Visit input iterables in a cycle until each is exhausted.

        >>> list(roundrobin('ABC', 'D', 'EF'))
        ['A', 'D', 'E', 'B', 'F', 'C']

    This function produces the same output as :func:`interleave_longest`, but
    may perform better for some inputs (in particular when the number of
    iterables is small).

    """
    # Algorithm credited to George Sakkis
    iterators = map(iter, iterables)
    for num_active in range(len(iterables), 0, -1):
        iterators = cycle(islice(iterators, num_active))
        yield from map(next, iterators)


def partition(pred, iterable):
    """
    Returns a 2-tuple of iterables derived from the input iterable.
    The first yields the items that have ``pred(item) == False``.
    The second yields the items that have ``pred(item) == True``.

        >>> is_odd = lambda x: x % 2 != 0
        >>> iterable = range(10)
        >>> even_items, odd_items = partition(is_odd, iterable)
        >>> list(even_items), list(odd_items)
        ([0, 2, 4, 6, 8], [1, 3, 5, 7, 9])

    If *pred* is None, :func:`bool` is used.

        >>> iterable = [0, 1, False, True, '', ' ']
        >>> false_items, true_items = partition(None, iterable)
        >>> list(false_items), list(true_items)
        ([0, False, ''], [1, True, ' '])

    """
    if pred is None:
        pred = bool

    t1, t2, p = tee(iterable, 3)
    p1, p2 = tee(map(pred, p))
    return (compress(t1, map(not_, p1)), compress(t2, p2))


def powerset(iterable):
    """Yields all possible subsets of the iterable.

        >>> list(powerset([1, 2, 3]))
        [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]

    :func:`powerset` will operate on iterables that aren't :class:`set`
    instances, so repeated elements in the input will produce repeated elements
    in the output.

        >>> seq = [1, 1, 0]
        >>> list(powerset(seq))
        [(), (1,), (1,), (0,), (1, 1), (1, 0), (1, 0), (1, 1, 0)]

    For a variant that efficiently yields actual :class:`set` instances, see
    :func:`powerset_of_sets`.
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def unique_everseen(iterable, key=None):
    """
    Yield unique elements, preserving order.

        >>> list(unique_everseen('AAAABBBCCDAABBB'))
        ['A', 'B', 'C', 'D']
        >>> list(unique_everseen('ABBCcAD', str.lower))
        ['A', 'B', 'C', 'D']

    Sequences with a mix of hashable and unhashable items can be used.
    The function will be slower (i.e., `O(n^2)`) for unhashable items.

    Remember that ``list`` objects are unhashable - you can use the *key*
    parameter to transform the list to a tuple (which is hashable) to
    avoid a slowdown.

        >>> iterable = ([1, 2], [2, 3], [1, 2])
        >>> list(unique_everseen(iterable))  # Slow
        [[1, 2], [2, 3]]
        >>> list(unique_everseen(iterable, key=tuple))  # Faster
        [[1, 2], [2, 3]]

    Similarly, you may want to convert unhashable ``set`` objects with
    ``key=frozenset``. For ``dict`` objects,
    ``key=lambda x: frozenset(x.items())`` can be used.

    """
    seenset = set()
    seenset_add = seenset.add
    seenlist = []
    seenlist_add = seenlist.append
    use_key = key is not None

    for element in iterable:
        k = key(element) if use_key else element
        try:
            if k not in seenset:
                seenset_add(k)
                yield element
        except TypeError:
            if k not in seenlist:
                seenlist_add(k)
                yield element


def unique_justseen(iterable, key=None):
    """Yields elements in order, ignoring serial duplicates

    >>> list(unique_justseen('AAAABBBCCDAABBB'))
    ['A', 'B', 'C', 'D', 'A', 'B']
    >>> list(unique_justseen('ABBCcAD', str.lower))
    ['A', 'B', 'C', 'A', 'D']

    """
    if key is None:
        return map(itemgetter(0), groupby(iterable))

    return map(next, map(itemgetter(1), groupby(iterable, key)))


def unique(iterable, key=None, reverse=False):
    """Yields unique elements in sorted order.

    >>> list(unique([[1, 2], [3, 4], [1, 2]]))
    [[1, 2], [3, 4]]

    *key* and *reverse* are passed to :func:`sorted`.

    >>> list(unique('ABBcCAD', str.casefold))
    ['A', 'B', 'c', 'D']
    >>> list(unique('ABBcCAD', str.casefold, reverse=True))
    ['D', 'c', 'B', 'A']

    The elements in *iterable* need not be hashable, but they must be
    comparable for sorting to work.
    """
    sequenced = sorted(iterable, key=key, reverse=reverse)
    return unique_justseen(sequenced, key=key)


def iter_except(func, exception, first=None):
    """Yields results from a function repeatedly until an exception is raised.

    Converts a call-until-exception interface to an iterator interface.
    Like ``iter(func, sentinel)``, but uses an exception instead of a sentinel
    to end the loop.

        >>> l = [0, 1, 2]
        >>> list(iter_except(l.pop, IndexError))
        [2, 1, 0]

    Multiple exceptions can be specified as a stopping condition:

        >>> l = [1, 2, 3, '...', 4, 5, 6]
        >>> list(iter_except(lambda: 1 + l.pop(), (IndexError, TypeError)))
        [7, 6, 5]
        >>> list(iter_except(lambda: 1 + l.pop(), (IndexError, TypeError)))
        [4, 3, 2]
        >>> list(iter_except(lambda: 1 + l.pop(), (IndexError, TypeError)))
        []

    """
    with suppress(exception):
        if first is not None:
            yield first()
        while True:
            yield func()


def first_true(iterable, default=None, pred=None):
    """
    Returns the first true value in the iterable.

    If no true value is found, returns *default*

    If *pred* is not None, returns the first item for which
    ``pred(item) == True`` .

        >>> first_true(range(10))
        1
        >>> first_true(range(10), pred=lambda x: x > 5)
        6
        >>> first_true(range(10), default='missing', pred=lambda x: x > 9)
        'missing'

    """
    return next(filter(pred, iterable), default)


def random_product(*args, repeat=1):
    """Draw an item at random from each of the input iterables.

        >>> random_product('abc', range(4), 'XYZ')  # doctest:+SKIP
        ('c', 3, 'Z')

    If *repeat* is provided as a keyword argument, that many items will be
    drawn from each iterable.

        >>> random_product('abcd', range(4), repeat=2)  # doctest:+SKIP
        ('a', 2, 'd', 3)

    This equivalent to taking a random selection from
    ``itertools.product(*args, repeat=repeat)``.

    """
    pools = [tuple(pool) for pool in args] * repeat
    return tuple(choice(pool) for pool in pools)


def random_permutation(iterable, r=None):
    """Return a random *r* length permutation of the elements in *iterable*.

    If *r* is not specified or is ``None``, then *r* defaults to the length of
    *iterable*.

        >>> random_permutation(range(5))  # doctest:+SKIP
        (3, 4, 0, 1, 2)

    This equivalent to taking a random selection from
    ``itertools.permutations(iterable, r)``.

    """
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return tuple(sample(pool, r))


def random_combination(iterable, r):
    """Return a random *r* length subsequence of the elements in *iterable*.

        >>> random_combination(range(5), 3)  # doctest:+SKIP
        (2, 3, 4)

    This equivalent to taking a random selection from
    ``itertools.combinations(iterable, r)``.

    """
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(sample(range(n), r))
    return tuple(pool[i] for i in indices)


def random_combination_with_replacement(iterable, r):
    """Return a random *r* length subsequence of elements in *iterable*,
    allowing individual elements to be repeated.

        >>> random_combination_with_replacement(range(3), 5) # doctest:+SKIP
        (0, 0, 1, 2, 2)

    This equivalent to taking a random selection from
    ``itertools.combinations_with_replacement(iterable, r)``.

    """
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(randrange(n) for i in range(r))
    return tuple(pool[i] for i in indices)


def nth_combination(iterable, r, index):
    """Equivalent to ``list(combinations(iterable, r))[index]``.

    The subsequences of *iterable* that are of length *r* can be ordered
    lexicographically. :func:`nth_combination` computes the subsequence at
    sort position *index* directly, without computing the previous
    subsequences.

        >>> nth_combination(range(5), 3, 5)
        (0, 3, 4)

    ``ValueError`` will be raised If *r* is negative or greater than the length
    of *iterable*.
    ``IndexError`` will be raised if the given *index* is invalid.
    """
    pool = tuple(iterable)
    n = len(pool)
    if (r < 0) or (r > n):
        raise ValueError

    c = 1
    k = min(r, n - r)
    for i in range(1, k + 1):
        c = c * (n - k + i) // i

    if index < 0:
        index += c

    if (index < 0) or (index >= c):
        raise IndexError

    result = []
    while r:
        c, n, r = c * r // n, n - 1, r - 1
        while index >= c:
            index -= c
            c, n = c * (n - r) // n, n - 1
        result.append(pool[-1 - n])

    return tuple(result)


def prepend(value, iterator):
    """Yield *value*, followed by the elements in *iterator*.

        >>> value = '0'
        >>> iterator = ['1', '2', '3']
        >>> list(prepend(value, iterator))
        ['0', '1', '2', '3']

    To prepend multiple values, see :func:`itertools.chain`
    or :func:`value_chain`.

    """
    return chain([value], iterator)


def convolve(signal, kernel):
    """Discrete linear convolution of two iterables.
    Equivalent to polynomial multiplication.

    For example, multiplying ``(x² -x - 20)`` by ``(x - 3)``
    gives ``(x³ -4x² -17x + 60)``.

        >>> list(convolve([1, -1, -20], [1, -3]))
        [1, -4, -17, 60]

    Examples of popular kinds of kernels:

    * The kernel ``[0.25, 0.25, 0.25, 0.25]`` computes a moving average.
      For image data, this blurs the image and reduces noise.
    * The kernel ``[1/2, 0, -1/2]`` estimates the first derivative of
      a function evaluated at evenly spaced inputs.
    * The kernel ``[1, -2, 1]`` estimates the second derivative of a
      function evaluated at evenly spaced inputs.

    Convolutions are mathematically commutative; however, the inputs are
    evaluated differently.  The signal is consumed lazily and can be
    infinite. The kernel is fully consumed before the calculations begin.

    Supports all numeric types: int, float, complex, Decimal, Fraction.

    References:

    * Article:  https://betterexplained.com/articles/intuitive-convolution/
    * Video by 3Blue1Brown:  https://www.youtube.com/watch?v=KuXjwB4LzSA

    """
    # This implementation comes from an older version of the itertools
    # documentation.  While the newer implementation is a bit clearer,
    # this one was kept because the inlined window logic is faster
    # and it avoids an unnecessary deque-to-tuple conversion.
    kernel = tuple(kernel)[::-1]
    n = len(kernel)
    window = deque([0], maxlen=n) * n
    for x in chain(signal, repeat(0, n - 1)):
        window.append(x)
        yield _sumprod(kernel, window)


def before_and_after(predicate, it):
    """A variant of :func:`takewhile` that allows complete access to the
    remainder of the iterator.

         >>> it = iter('ABCdEfGhI')
         >>> all_upper, remainder = before_and_after(str.isupper, it)
         >>> ''.join(all_upper)
         'ABC'
         >>> ''.join(remainder) # takewhile() would lose the 'd'
         'dEfGhI'

    Note that the first iterator must be fully consumed before the second
    iterator can generate valid results.
    """
    trues, after = tee(it)
    trues = compress(takewhile(predicate, trues), zip(after))
    return trues, after


def triplewise(iterable):
    """Return overlapping triplets from *iterable*.

    >>> list(triplewise('ABCDE'))
    [('A', 'B', 'C'), ('B', 'C', 'D'), ('C', 'D', 'E')]

    """
    # This deviates from the itertools documentation recipe - see
    # https://github.com/more-itertools/more-itertools/issues/889
    t1, t2, t3 = tee(iterable, 3)
    next(t3, None)
    next(t3, None)
    next(t2, None)
    return zip(t1, t2, t3)


def _sliding_window_islice(iterable, n):
    # Fast path for small, non-zero values of n.
    iterators = tee(iterable, n)
    for i, iterator in enumerate(iterators):
        next(islice(iterator, i, i), None)
    return zip(*iterators)


def _sliding_window_deque(iterable, n):
    # Normal path for other values of n.
    iterator = iter(iterable)
    window = deque(islice(iterator, n - 1), maxlen=n)
    for x in iterator:
        window.append(x)
        yield tuple(window)


def sliding_window(iterable, n):
    """Return a sliding window of width *n* over *iterable*.

        >>> list(sliding_window(range(6), 4))
        [(0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5)]

    If *iterable* has fewer than *n* items, then nothing is yielded:

        >>> list(sliding_window(range(3), 4))
        []

    For a variant with more features, see :func:`windowed`.
    """
    if n > 20:
        return _sliding_window_deque(iterable, n)
    elif n > 2:
        return _sliding_window_islice(iterable, n)
    elif n == 2:
        return pairwise(iterable)
    elif n == 1:
        return zip(iterable)
    else:
        raise ValueError(f'n should be at least one, not {n}')


def subslices(iterable):
    """Return all contiguous non-empty subslices of *iterable*.

        >>> list(subslices('ABC'))
        [['A'], ['A', 'B'], ['A', 'B', 'C'], ['B'], ['B', 'C'], ['C']]

    This is similar to :func:`substrings`, but emits items in a different
    order.
    """
    seq = list(iterable)
    slices = starmap(slice, combinations(range(len(seq) + 1), 2))
    return map(getitem, repeat(seq), slices)


def polynomial_from_roots(roots):
    """Compute a polynomial's coefficients from its roots.

    >>> roots = [5, -4, 3]            # (x - 5) * (x + 4) * (x - 3)
    >>> polynomial_from_roots(roots)  # x³ - 4 x² - 17 x + 60
    [1, -4, -17, 60]

    Note that polynomial coefficients are specified in descending power order.

    Supports all numeric types: int, float, complex, Decimal, Fraction.
    """

    # This recipe differs from the one in itertools docs in that it
    # applies list() after each call to convolve().  This avoids
    # hitting stack limits with nested generators.

    poly = [1]
    for root in roots:
        poly = list(convolve(poly, (1, -root)))
    return poly


def iter_index(iterable, value, start=0, stop=None):
    """Yield the index of each place in *iterable* that *value* occurs,
    beginning with index *start* and ending before index *stop*.


    >>> list(iter_index('AABCADEAF', 'A'))
    [0, 1, 4, 7]
    >>> list(iter_index('AABCADEAF', 'A', 1))  # start index is inclusive
    [1, 4, 7]
    >>> list(iter_index('AABCADEAF', 'A', 1, 7))  # stop index is not inclusive
    [1, 4]

    The behavior for non-scalar *values* matches the built-in Python types.

    >>> list(iter_index('ABCDABCD', 'AB'))
    [0, 4]
    >>> list(iter_index([0, 1, 2, 3, 0, 1, 2, 3], [0, 1]))
    []
    >>> list(iter_index([[0, 1], [2, 3], [0, 1], [2, 3]], [0, 1]))
    [0, 2]

    See :func:`locate` for a more general means of finding the indexes
    associated with particular values.

    """
    seq_index = getattr(iterable, 'index', None)
    if seq_index is None:
        # Slow path for general iterables
        iterator = islice(iterable, start, stop)
        for i, element in enumerate(iterator, start):
            if element is value or element == value:
                yield i
    else:
        # Fast path for sequences
        stop = len(iterable) if stop is None else stop
        i = start - 1
        with suppress(ValueError):
            while True:
                yield (i := seq_index(value, i + 1, stop))


def sieve(n):
    """Yield the primes less than n.

    >>> list(sieve(30))
    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

    """
    # This implementation comes from an older version of the itertools
    # documentation.  The newer implementation is easier to read but is
    # less lazy.
    if n > 2:
        yield 2
    start = 3
    data = bytearray((0, 1)) * (n // 2)
    for p in iter_index(data, 1, start, stop=isqrt(n) + 1):
        yield from iter_index(data, 1, start, p * p)
        data[p * p : n : p + p] = bytes(len(range(p * p, n, p + p)))
        start = p * p
    yield from iter_index(data, 1, start)


def _batched(iterable, n, *, strict=False):  # pragma: no cover
    """Batch data into tuples of length *n*. If the number of items in
    *iterable* is not divisible by *n*:
    * The last batch will be shorter if *strict* is ``False``.
    * :exc:`ValueError` will be raised if *strict* is ``True``.

    >>> list(batched('ABCDEFG', 3))
    [('A', 'B', 'C'), ('D', 'E', 'F'), ('G',)]

    On Python 3.13 and above, this is an alias for :func:`itertools.batched`.
    """
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError('batched(): incomplete batch')
        yield batch


if hexversion >= 0x30D00A2:  # pragma: no cover
    from itertools import batched as itertools_batched

    def batched(iterable, n, *, strict=False):
        return itertools_batched(iterable, n, strict=strict)

    batched.__doc__ = _batched.__doc__
else:  # pragma: no cover
    batched = _batched


def transpose(it):
    """Swap the rows and columns of the input matrix.

    >>> list(transpose([(1, 2, 3), (11, 22, 33)]))
    [(1, 11), (2, 22), (3, 33)]

    The caller should ensure that the dimensions of the input are compatible.
    If the input is empty, no output will be produced.
    """
    return _zip_strict(*it)


def _is_scalar(value, stringlike=(str, bytes)):
    "Scalars are bytes, strings, and non-iterables."
    try:
        iter(value)
    except TypeError:
        return True
    return isinstance(value, stringlike)


def _flatten_tensor(tensor):
    "Depth-first iterator over scalars in a tensor."
    iterator = iter(tensor)
    while True:
        try:
            value = next(iterator)
        except StopIteration:
            return iterator
        iterator = chain((value,), iterator)
        if _is_scalar(value):
            return iterator
        iterator = chain.from_iterable(iterator)


def reshape(matrix, shape):
    """Change the shape of a *matrix*.

    If *shape* is an integer, the matrix must be two dimensional
    and the shape is interpreted as the desired number of columns:

        >>> matrix = [(0, 1), (2, 3), (4, 5)]
        >>> cols = 3
        >>> list(reshape(matrix, cols))
        [(0, 1, 2), (3, 4, 5)]

    If *shape* is a tuple (or other iterable), the input matrix can have
    any number of dimensions. It will first be flattened and then rebuilt
    to the desired shape which can also be multidimensional:

        >>> matrix = [(0, 1), (2, 3), (4, 5)]    # Start with a 3 x 2 matrix

        >>> list(reshape(matrix, (2, 3)))        # Make a 2 x 3 matrix
        [(0, 1, 2), (3, 4, 5)]

        >>> list(reshape(matrix, (6,)))          # Make a vector of length six
        [0, 1, 2, 3, 4, 5]

        >>> list(reshape(matrix, (2, 1, 3, 1)))  # Make 2 x 1 x 3 x 1 tensor
        [(((0,), (1,), (2,)),), (((3,), (4,), (5,)),)]

    Each dimension is assumed to be uniform, either all arrays or all scalars.
    Flattening stops when the first value in a dimension is a scalar.
    Scalars are bytes, strings, and non-iterables.
    The reshape iterator stops when the requested shape is complete
    or when the input is exhausted, whichever comes first.

    """
    if isinstance(shape, int):
        return batched(chain.from_iterable(matrix), shape)
    first_dim, *dims = shape
    scalar_stream = _flatten_tensor(matrix)
    reshaped = reduce(batched, reversed(dims), scalar_stream)
    return islice(reshaped, first_dim)


def matmul(m1, m2):
    """Multiply two matrices.

    >>> list(matmul([(7, 5), (3, 5)], [(2, 5), (7, 9)]))
    [(49, 80), (41, 60)]

    The caller should ensure that the dimensions of the input matrices are
    compatible with each other.

    Supports all numeric types: int, float, complex, Decimal, Fraction.
    """
    n = len(m2[0])
    return batched(starmap(_sumprod, product(m1, transpose(m2))), n)


def _factor_pollard(n):
    # Return a factor of n using Pollard's rho algorithm.
    # Efficient when n is odd and composite.
    for b in range(1, n):
        x = y = 2
        d = 1
        while d == 1:
            x = (x * x + b) % n
            y = (y * y + b) % n
            y = (y * y + b) % n
            d = gcd(x - y, n)
        if d != n:
            return d
    raise ValueError('prime or under 5')  # pragma: no cover


_primes_below_211 = tuple(sieve(211))


def factor(n):
    """Yield the prime factors of n.

    >>> list(factor(360))
    [2, 2, 2, 3, 3, 5]

    Finds small factors with trial division.  Larger factors are
    either verified as prime with ``is_prime`` or split into
    smaller factors with Pollard's rho algorithm.
    """

    # Corner case reduction
    if n < 2:
        return

    # Trial division reduction
    for prime in _primes_below_211:
        while not n % prime:
            yield prime
            n //= prime

    # Pollard's rho reduction
    primes = []
    todo = [n] if n > 1 else []
    for n in todo:
        if n < 211**2 or is_prime(n):
            primes.append(n)
        else:
            fact = _factor_pollard(n)
            todo += (fact, n // fact)
    yield from sorted(primes)


def polynomial_eval(coefficients, x):
    """Evaluate a polynomial at a specific value.

    Computes with better numeric stability than Horner's method.

    Evaluate ``x^3 - 4 * x^2 - 17 * x + 60`` at ``x = 2.5``:

    >>> coefficients = [1, -4, -17, 60]
    >>> x = 2.5
    >>> polynomial_eval(coefficients, x)
    8.125

    Note that polynomial coefficients are specified in descending power order.

    Supports all numeric types: int, float, complex, Decimal, Fraction.
    """
    n = len(coefficients)
    if n == 0:
        return type(x)(0)
    powers = map(pow, repeat(x), reversed(range(n)))
    return _sumprod(coefficients, powers)


def sum_of_squares(it):
    """Return the sum of the squares of the input values.

    >>> sum_of_squares([10, 20, 30])
    1400

    Supports all numeric types: int, float, complex, Decimal, Fraction.
    """
    return _sumprod(*tee(it))


def polynomial_derivative(coefficients):
    """Compute the first derivative of a polynomial.

    Evaluate the derivative of ``x³ - 4 x² - 17 x + 60``:

    >>> coefficients = [1, -4, -17, 60]
    >>> derivative_coefficients = polynomial_derivative(coefficients)
    >>> derivative_coefficients
    [3, -8, -17]

    Note that polynomial coefficients are specified in descending power order.

    Supports all numeric types: int, float, complex, Decimal, Fraction.
    """
    n = len(coefficients)
    powers = reversed(range(1, n))
    return list(map(mul, coefficients, powers))


def totient(n):
    """Return the count of natural numbers up to *n* that are coprime with *n*.

    Euler's totient function φ(n) gives the number of totatives.
    Totative are integers k in the range 1 ≤ k ≤ n such that gcd(n, k) = 1.

    >>> n = 9
    >>> totient(n)
    6

    >>> totatives = [x for x in range(1, n) if gcd(n, x) == 1]
    >>> totatives
    [1, 2, 4, 5, 7, 8]
    >>> len(totatives)
    6

    Reference:  https://en.wikipedia.org/wiki/Euler%27s_totient_function

    """
    for prime in set(factor(n)):
        n -= n // prime
    return n


# Miller–Rabin primality test: https://oeis.org/A014233
_perfect_tests = [
    (2047, (2,)),
    (9080191, (31, 73)),
    (4759123141, (2, 7, 61)),
    (1122004669633, (2, 13, 23, 1662803)),
    (2152302898747, (2, 3, 5, 7, 11)),
    (3474749660383, (2, 3, 5, 7, 11, 13)),
    (18446744073709551616, (2, 325, 9375, 28178, 450775, 9780504, 1795265022)),
    (
        3317044064679887385961981,
        (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41),
    ),
]


@lru_cache
def _shift_to_odd(n):
    'Return s, d such that 2**s * d == n'
    s = ((n - 1) ^ n).bit_length() - 1
    d = n >> s
    assert (1 << s) * d == n and d & 1 and s >= 0
    return s, d


def _strong_probable_prime(n, base):
    assert (n > 2) and (n & 1) and (2 <= base < n)

    s, d = _shift_to_odd(n - 1)

    x = pow(base, d, n)
    if x == 1 or x == n - 1:
        return True

    for _ in range(s - 1):
        x = x * x % n
        if x == n - 1:
            return True

    return False


# Separate instance of Random() that doesn't share state
# with the default user instance of Random().
_private_randrange = random.Random().randrange


def is_prime(n):
    """Return ``True`` if *n* is prime and ``False`` otherwise.

    Basic examples:

        >>> is_prime(37)
        True
        >>> is_prime(3 * 13)
        False
        >>> is_prime(18_446_744_073_709_551_557)
        True

    Find the next prime over one billion:

        >>> next(filter(is_prime, count(10**9)))
        1000000007

    Generate random primes up to 200 bits and up to 60 decimal digits:

        >>> from random import seed, randrange, getrandbits
        >>> seed(18675309)

        >>> next(filter(is_prime, map(getrandbits, repeat(200))))
        893303929355758292373272075469392561129886005037663238028407

        >>> next(filter(is_prime, map(randrange, repeat(10**60))))
        269638077304026462407872868003560484232362454342414618963649

    This function is exact for values of *n* below 10**24.  For larger inputs,
    the probabilistic Miller-Rabin primality test has a less than 1 in 2**128
    chance of a false positive.
    """

    if n < 17:
        return n in {2, 3, 5, 7, 11, 13}

    if not (n & 1 and n % 3 and n % 5 and n % 7 and n % 11 and n % 13):
        return False

    for limit, bases in _perfect_tests:
        if n < limit:
            break
    else:
        bases = (_private_randrange(2, n - 1) for i in range(64))

    return all(_strong_probable_prime(n, base) for base in bases)


def loops(n):
    """Returns an iterable with *n* elements for efficient looping.
    Like ``range(n)`` but doesn't create integers.

    >>> i = 0
    >>> for _ in loops(5):
    ...     i += 1
    >>> i
    5

    """
    return repeat(None, n)


def multinomial(*counts):
    """Number of distinct arrangements of a multiset.

    The expression ``multinomial(3, 4, 2)`` has several equivalent
    interpretations:

    * In the expansion of ``(a + b + c)⁹``, the coefficient of the
      ``a³b⁴c²`` term is 1260.

    * There are 1260 distinct ways to arrange 9 balls consisting of 3 reds, 4
      greens, and 2 blues.

    * There are 1260 unique ways to place 9 distinct objects into three bins
      with sizes 3, 4, and 2.

    The :func:`multinomial` function computes the length of
    :func:`distinct_permutations`.  For example, there are 83,160 distinct
    anagrams of the word "abracadabra":

        >>> from more_itertools import distinct_permutations, ilen
        >>> ilen(distinct_permutations('abracadabra'))
        83160

    This can be computed directly from the letter counts, 5a 2b 2r 1c 1d:

        >>> from collections import Counter
        >>> list(Counter('abracadabra').values())
        [5, 2, 2, 1, 1]
        >>> multinomial(5, 2, 2, 1, 1)
        83160

    A binomial coefficient is a special case of multinomial where there are
    only two categories.  For example, the number of ways to arrange 12 balls
    with 5 reds and 7 blues is ``multinomial(5, 7)`` or ``math.comb(12, 5)``.

    Likewise, factorial is a special case of multinomial where
    the multiplicities are all just 1 so that
    ``multinomial(1, 1, 1, 1, 1, 1, 1) == math.factorial(7)``.

    Reference:  https://en.wikipedia.org/wiki/Multinomial_theorem

    """
    return prod(map(comb, accumulate(counts), counts))


def _running_median_minheap_and_maxheap(iterator):  # pragma: no cover
    "Non-windowed running_median() for Python 3.14+"

    read = iterator.__next__
    lo = []  # max-heap
    hi = []  # min-heap (same size as or one smaller than lo)

    with suppress(StopIteration):
        while True:
            heappush_max(lo, heappushpop(hi, read()))
            yield lo[0]

            heappush(hi, heappushpop_max(lo, read()))
            yield (lo[0] + hi[0]) / 2


def _running_median_minheap_only(iterator):  # pragma: no cover
    "Backport of non-windowed running_median() for Python 3.13 and prior."

    read = iterator.__next__
    lo = []  # max-heap (actually a minheap with negated values)
    hi = []  # min-heap (same size as or one smaller than lo)

    with suppress(StopIteration):
        while True:
            heappush(lo, -heappushpop(hi, read()))
            yield -lo[0]

            heappush(hi, -heappushpop(lo, -read()))
            yield (hi[0] - lo[0]) / 2


def _running_median_windowed(iterator, maxlen):
    "Yield median of values in a sliding window."

    window = deque()
    ordered = []

    for x in iterator:
        window.append(x)
        insort(ordered, x)

        if len(ordered) > maxlen:
            i = bisect_left(ordered, window.popleft())
            del ordered[i]

        n = len(ordered)
        m = n // 2
        yield ordered[m] if n & 1 else (ordered[m - 1] + ordered[m]) / 2


def running_median(iterable, *, maxlen=None):
    """Cumulative median of values seen so far or values in a sliding window.

    Set *maxlen* to a positive integer to specify the maximum size
    of the sliding window.  The default of *None* is equivalent to
    an unbounded window.

    For example:

        >>> list(running_median([5.0, 9.0, 4.0, 12.0, 8.0, 9.0]))
        [5.0, 7.0, 5.0, 7.0, 8.0, 8.5]
        >>> list(running_median([5.0, 9.0, 4.0, 12.0, 8.0, 9.0], maxlen=3))
        [5.0, 7.0, 5.0, 9.0, 8.0, 9.0]

    Supports numeric types such as int, float, Decimal, and Fraction,
    but not complex numbers which are unorderable.

    On version Python 3.13 and prior, max-heaps are simulated with
    negative values. The negation causes Decimal inputs to apply context
    rounding, making the results slightly different than that obtained
    by statistics.median().
    """

    iterator = iter(iterable)

    if maxlen is not None:
        maxlen = index(maxlen)
        if maxlen <= 0:
            raise ValueError('Window size should be positive')
        return _running_median_windowed(iterator, maxlen)

    if not _max_heap_available:
        return _running_median_minheap_only(iterator)  # pragma: no cover

    return _running_median_minheap_and_maxheap(iterator)  # pragma: no cover

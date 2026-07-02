# Copyright 2006 Google, Inc. All Rights Reserved.
# Licensed to PSF under a Contributor Agreement.

"""Fixer for has_key().

Calls to .has_key() methods are expressed in terms of the 'in'
operator:

    d.has_key(k) -> k in d

CAVEATS:
1) While the primary target of this fixer is dict.has_key(), the
   fixer will change any has_key() method call, regardless of its
   class.

2) Cases like this will not be converted:

    m = d.has_key
    if m(k):
        ...

   Only *calls* to has_key() are converted. While it is possible to
   convert the above to something like

    m = d.__contains__
    if m(k):
        ...

   this is currently not done.
"""

# Local imports
from .. import pytree
from .. import fixer_base
from ..fixer_util import Name, parenthesize


class FixHasKey(fixer_base.BaseFix):
    BM_compatible = True

    PATTERN = """
    anchor=power<
        before=any+
        trailer< '.' 'has_key' >
        trailer<
            '('
            ( not(arglist | argument<any '=' any>) arg=any
            | arglist<(not argument<any '=' any>) arg=any ','>
            )
            ')'
        >
        after=any*
    >
    |
    negation=not_test<
        'not'
        anchor=power<
            before=any+
            trailer< '.' 'has_key' >
            trailer<
                '('
                ( not(arglist | argument<any '=' any>) arg=any
                | arglist<(not argument<any '=' any>) arg=any ','>
                )
                ')'
            >
        >
    >
    """

    def transform(self, node, results):
        assert results
        syms = self.syms
        if (node.parent.type == syms.not_test and
            self.pattern.match(node.parent)):
            # Don't transform a node matching the first alternative of the
            # pattern when its parent matches the second alternative
            return None
        negation = results.get("negation")
        anchor = results["anchor"]
        prefix = node.prefix
        before = [n.clone() for n in results["before"]]
        arg = results["arg"].clone()
        after = results.get("after")
        if after:
            after = [n.clone() for n in after]
        if arg.type in (syms.comparison, syms.not_test, syms.and_test,
                        syms.or_test, syms.test, syms.lambdef, syms.argument):
            arg = parenthesize(arg)
        if len(before) == 1:
            before = before[0]
        else:
            before = pytree.Node(syms.power, before)
        before.prefix = " "
        n_op = Name("in", prefix=" ")
        if negation:
            n_not = Name("not", prefix=" ")
            n_op = pytree.Node(syms.comp_op, (n_not, n_op))
        new = pytree.Node(syms.comparison, (arg, n_op, before))
        if after:
            new = parenthesize(new)
            new = pytree.Node(syms.power, (new,) + tuple(after))
        if node.parent.type in (syms.comparison, syms.expr, syms.xor_expr,
                                syms.and_expr, syms.shift_expr,
                                syms.arith_expr, syms.term,
                                syms.factor, syms.power):
            new = parenthesize(new)
        new.prefix = prefix
        return new

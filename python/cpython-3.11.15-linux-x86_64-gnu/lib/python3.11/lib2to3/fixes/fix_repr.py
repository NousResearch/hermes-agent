# Copyright 2006 Google, Inc. All Rights Reserved.
# Licensed to PSF under a Contributor Agreement.

"""Fixer that transforms `xyzzy` into repr(xyzzy)."""

# Local imports
from .. import fixer_base
from ..fixer_util import Call, Name, parenthesize


class FixRepr(fixer_base.BaseFix):

    BM_compatible = True
    PATTERN = """
              atom < '`' expr=any '`' >
              """

    def transform(self, node, results):
        expr = results["expr"].clone()

        if expr.type == self.syms.testlist1:
            expr = parenthesize(expr)
        return Call(Name("repr"), [expr], prefix=node.prefix)

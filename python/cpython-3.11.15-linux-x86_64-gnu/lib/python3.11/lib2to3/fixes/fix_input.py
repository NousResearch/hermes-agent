"""Fixer that changes input(...) into eval(input(...))."""
# Author: Andre Roberge

# Local imports
from .. import fixer_base
from ..fixer_util import Call, Name
from .. import patcomp


context = patcomp.compile_pattern("power< 'eval' trailer< '(' any ')' > >")


class FixInput(fixer_base.BaseFix):
    BM_compatible = True
    PATTERN = """
              power< 'input' args=trailer< '(' [any] ')' > >
              """

    def transform(self, node, results):
        # If we're already wrapped in an eval() call, we're done.
        if context.match(node.parent.parent):
            return

        new = node.clone()
        new.prefix = ""
        return Call(Name("eval"), [new], prefix=node.prefix)

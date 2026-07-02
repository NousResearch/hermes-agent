"""
Fixer that changes zip(seq0, seq1, ...) into list(zip(seq0, seq1, ...)
unless there exists a 'from future_builtins import zip' statement in the
top-level namespace.

We avoid the transformation if the zip() call is directly contained in
iter(<>), list(<>), tuple(<>), sorted(<>), ...join(<>), or for V in <>:.
"""

# Local imports
from .. import fixer_base
from ..pytree import Node
from ..pygram import python_symbols as syms
from ..fixer_util import Name, ArgList, in_special_context


class FixZip(fixer_base.ConditionalFix):

    BM_compatible = True
    PATTERN = """
    power< 'zip' args=trailer< '(' [any] ')' > [trailers=trailer*]
    >
    """

    skip_on = "future_builtins.zip"

    def transform(self, node, results):
        if self.should_skip(node):
            return

        if in_special_context(node):
            return None

        args = results['args'].clone()
        args.prefix = ""

        trailers = []
        if 'trailers' in results:
            trailers = [n.clone() for n in results['trailers']]
            for n in trailers:
                n.prefix = ""

        new = Node(syms.power, [Name("zip"), args], prefix="")
        new = Node(syms.power, [Name("list"), ArgList([new])] + trailers)
        new.prefix = node.prefix
        return new

"""Fixer for __nonzero__ -> __bool__ methods."""
# Author: Collin Winter

# Local imports
from .. import fixer_base
from ..fixer_util import Name

class FixNonzero(fixer_base.BaseFix):
    BM_compatible = True
    PATTERN = """
    classdef< 'class' any+ ':'
              suite< any*
                     funcdef< 'def' name='__nonzero__'
                              parameters< '(' NAME ')' > any+ >
                     any* > >
    """

    def transform(self, node, results):
        name = results["name"]
        new = Name("__bool__", prefix=name.prefix)
        name.replace(new)

"""Fixer for basestring -> str."""
# Author: Christian Heimes

# Local imports
from .. import fixer_base
from ..fixer_util import Name

class FixBasestring(fixer_base.BaseFix):
    BM_compatible = True

    PATTERN = "'basestring'"

    def transform(self, node, results):
        return Name("str", prefix=node.prefix)

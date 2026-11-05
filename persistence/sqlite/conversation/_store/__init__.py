"""The legacy store's conversation SQL, run by the conversation repositories."""

# The store modules import one another at their ends; entered through compression that order resolves every name,
# entered through sessions it does not (prune_rewrite needs a sessions name not yet defined), so load it first.
from . import compression

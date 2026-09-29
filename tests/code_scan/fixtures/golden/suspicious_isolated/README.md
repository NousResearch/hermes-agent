# Suspicious Isolated Fixture

This fixture contains a single standalone Python source file
with no project structure, no imports, and no references
from other files.  A UA scan should detect it as suspicious
and report it as an orphan/unreachable node.

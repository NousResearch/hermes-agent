import sys

if sys.version_info >= (3, 10):
    from test.support.import_helper import (
        CleanImport as CleanImport,
    )
    from test.support.import_helper import (
        DirsOnSysPath as DirsOnSysPath,
    )
    from test.support.os_helper import (
        EnvironmentVarGuard as EnvironmentVarGuard,
    )
    from test.support.os_helper import (
        rmtree as rmtree,
    )
    from test.support.os_helper import (
        skip_unless_symlink as skip_unless_symlink,
    )
    from test.support.os_helper import (
        unlink as unlink,
    )
else:
    from test.support import (
        CleanImport as CleanImport,
    )
    from test.support import (
        DirsOnSysPath as DirsOnSysPath,
    )
    from test.support import (
        EnvironmentVarGuard as EnvironmentVarGuard,
    )
    from test.support import (
        rmtree as rmtree,
    )
    from test.support import (
        skip_unless_symlink as skip_unless_symlink,
    )
    from test.support import (
        unlink as unlink,
    )

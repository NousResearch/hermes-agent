# pysqlite2/__init__.py: the pysqlite2 package.
#
# Copyright (C) 2005 Gerhard Häring <gh@ghaering.de>
#
# This file is part of pysqlite.
#
# This software is provided 'as-is', without any express or implied
# warranty.  In no event will the authors be held liable for any damages
# arising from the use of this software.
#
# Permission is granted to anyone to use this software for any purpose,
# including commercial applications, and to alter it and redistribute it
# freely, subject to the following restrictions:
#
# 1. The origin of this software must not be misrepresented; you must not
#    claim that you wrote the original software. If you use this software
#    in a product, an acknowledgment in the product documentation would be
#    appreciated but is not required.
# 2. Altered source versions must be plainly marked as such, and must not be
#    misrepresented as being the original software.
# 3. This notice may not be removed or altered from any source distribution.

"""
The sqlite3 extension module provides a DB-API 2.0 (PEP 249) compliant
interface to the SQLite library, and requires SQLite 3.7.15 or newer.

To use the module, start by creating a database Connection object:

    import sqlite3
    cx = sqlite3.connect("test.db")  # test.db will be created or opened

The special path name ":memory:" can be provided to connect to a transient
in-memory database:

    cx = sqlite3.connect(":memory:")  # connect to a database in RAM

Once a connection has been established, create a Cursor object and call
its execute() method to perform SQL queries:

    cu = cx.cursor()

    # create a table
    cu.execute("create table lang(name, first_appeared)")

    # insert values into a table
    cu.execute("insert into lang values (?, ?)", ("C", 1972))

    # execute a query and iterate over the result
    for row in cu.execute("select * from lang"):
        print(row)

    cx.close()

The sqlite3 module is written by Gerhard Häring <gh@ghaering.de>.
"""

from sqlite3.dbapi2 import *


# bpo-42264: OptimizedUnicode was deprecated in Python 3.10.  It's scheduled
# for removal in Python 3.12.
def __getattr__(name):
    if name == "OptimizedUnicode":
        import warnings
        msg = ("""
            OptimizedUnicode is deprecated and will be removed in Python 3.12.
            Since Python 3.3 it has simply been an alias for 'str'.
        """)
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return str
    raise AttributeError(f"module 'sqlite3' has no attribute '{name}'")

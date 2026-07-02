# Copyright 2014-2016 Nathan West
#
# This file is part of autocommand.
#
# autocommand is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# autocommand is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with autocommand.  If not, see <http://www.gnu.org/licenses/>.

# flake8 flags all these imports as unused, hence the NOQAs everywhere.

from .automain import automain  # NOQA
from .autoparse import autoparse, smart_open  # NOQA
from .autocommand import autocommand  # NOQA

try:
    from .autoasync import autoasync  # NOQA
except ImportError:  # pragma: no cover
    pass

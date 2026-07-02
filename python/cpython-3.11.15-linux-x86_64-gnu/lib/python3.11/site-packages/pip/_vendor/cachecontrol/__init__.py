# SPDX-FileCopyrightText: 2015 Eric Larson
#
# SPDX-License-Identifier: Apache-2.0

"""CacheControl import Interface.

Make it easy to import from cachecontrol without long namespaces.
"""

import importlib.metadata

from pip._vendor.cachecontrol.adapter import CacheControlAdapter
from pip._vendor.cachecontrol.controller import CacheController
from pip._vendor.cachecontrol.wrapper import CacheControl

__author__ = "Eric Larson"
__email__ = "eric@ionrock.org"
# pip patch: this won't work when vendored, so just patch it out as it's unused
# __version__ = importlib.metadata.version("cachecontrol")

__all__ = [
    "__author__",
    "__email__",
    "__version__",
    "CacheControlAdapter",
    "CacheController",
    "CacheControl",
]

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

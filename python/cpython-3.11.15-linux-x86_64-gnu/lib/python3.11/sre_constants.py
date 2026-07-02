import warnings
warnings.warn(f"module {__name__!r} is deprecated",
              DeprecationWarning,
              stacklevel=2)

from re import _constants as _
globals().update({k: v for k, v in vars(_).items() if k[:2] != '__'})

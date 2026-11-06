"""Photon Spectrum (iMessage) platform plugin entry point."""
# Canonical name (not ``from .adapter``): the loader runs this package as ``hermes_plugins.platforms__photon``;
# anything importing ``plugins.platforms.photon.*`` would otherwise execute the adapter a second time.
from plugins.platforms.photon.adapter import register

__all__ = ["register"]

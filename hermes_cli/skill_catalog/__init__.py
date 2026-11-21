"""Experimental plugin-served catalogs.

``provider`` defines the synchronous plain-data contract. ``identity`` validates
immutable descriptors without running plugin attributes. Registry, service and
mutation modules own lifecycle, dispatch and cooperative filesystem admission.
"""

from hermes_cli.skill_catalog.provider import SkillCatalogProvider

__all__ = ["SkillCatalogProvider"]

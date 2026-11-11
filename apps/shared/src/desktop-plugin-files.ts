/** Catalog names are directory names under `plugin-catalog/`; anything else is not a lookup key. */
export const PLUGIN_CATALOG_NAME_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/

/** Marker inside a materialized desktop half: which agent package it came from. */
export const PACKAGE_MARKER = '.hermes-package.json'

/** Inside a STANDALONE desktop plugin installed from the catalog: which entry it is (`catalogName`, `repo`,
 *  `sha`, and `pin` when a tag pin names a tag object), so the catalog card shows it installed. Unlike the
 *  package marker it pairs the folder with no agent half and leaves it removable. Only an installer writes it
 *  (`hermes_cli/desktop_plugin_halves.py` `INSTALL_ORIGIN` on the Webapp host). */
export const INSTALL_ORIGIN = '.hermes-origin.json'

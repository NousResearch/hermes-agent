// tests/install/e2e-assets runs from the checkout but has no package.json of
// its own: its typed drivers reach the dependencies tests-js declares through
// this module, which resolves them from tests-js wherever the install put them.
export { _electron, type ElectronApplication, type Page } from '@playwright/test'
export { z } from 'zod'

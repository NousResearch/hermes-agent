// Renderer entry. The whole app (every store that reads localStorage when its
// module evaluates) enters through the dynamic import below, so nothing can
// read a shared preference before this machine's store hydrated it (#145213),
// in the production chunk graph and in dev's unbundled ESM alike. Keep this
// file's static imports free of app modules.
import { bootRenderer } from './lib/ui-preferences/boot'

void bootRenderer(() => import('./main'))

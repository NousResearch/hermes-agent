import type { HermesPlugin } from '@hermes/plugin-sdk'
import {
  host,
  KEYBINDS_AREA,
  PALETTE_AREA,
  ROUTES_AREA,
  SIDEBAR_NAV_AREA,
  STATUSBAR_AREAS,
} from '@hermes/plugin-sdk'

const FOCUS_KEY = 'unitalk-focus-mode'

function toggleFocus() {
  if (window.location.pathname === '/focus') {
    host.navigate('/')
  } else {
    host.navigate('/focus')
  }
}

const plugin: HermesPlugin = {
  id: 'focus-mode',
  name: 'Mode Bureau',
  description: "Chat plein écran sans distractions. L'essentiel, rien que l'essentiel.",
  defaultEnabled: false,

  register(ctx) {
    ctx.registerMany([
      {
        id: 'page',
        area: ROUTES_AREA,
        data: { path: '/focus' },
        render() {
          const modKey = navigator.platform.includes('Mac') ? '⌘' : 'Ctrl'
          return (
            <div style={{
              position: 'fixed', inset: 0, zIndex: 100,
              background: '#F3EFE6', display: 'flex', flexDirection: 'column',
              fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif',
            }}>
              <button
                type="button"
                onClick={() => host.navigate('/')}
                style={{
                  position: 'fixed', top: 16, right: 16, zIndex: 200,
                  display: 'flex', alignItems: 'center', gap: 6, padding: '8px 14px',
                  background: 'rgba(28,26,23,0.06)', border: '1px solid #D8D0C2',
                  borderRadius: 10, color: '#7A7468', fontSize: '0.8rem',
                  fontWeight: 500, cursor: 'pointer', fontFamily: 'inherit',
                  backdropFilter: 'blur(4px)',
                }}
              >
                ← Quitter le Bureau
              </button>
              <div style={{
                flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                <div style={{ textAlign: 'center', maxWidth: 400, padding: 24 }}>
                  <div style={{ fontSize: '2.5rem', marginBottom: 16 }}>🎯</div>
                  <h1 style={{ fontSize: '1.3rem', fontWeight: 700, color: '#1C1A17', margin: '0 0 8px' }}>
                    Mode Bureau activé
                  </h1>
                  <p style={{ fontSize: '0.9rem', color: '#7A7468', margin: '0 0 16px', lineHeight: 1.5 }}>
                    Ouvrez une conversation et votre Collaborateur est à votre écoute.<br />
                    L'essentiel, rien que l'essentiel.
                  </p>
                  <span style={{ fontSize: '0.75rem', color: '#B8B0A4' }}>
                    {modKey}+Shift+F pour quitter
                  </span>
                </div>
              </div>
            </div>
          )
        },
      },
      {
        id: 'nav',
        area: SIDEBAR_NAV_AREA,
        order: 900,
        data: { codicon: 'target', label: 'Bureau', path: '/focus' },
      },
      {
        id: 'statusbar',
        area: STATUSBAR_AREAS.right,
        render() {
          if (typeof window === 'undefined' || window.location.pathname !== '/focus') return null
          return <span style={{ fontSize: '0.69rem', color: '#D10E63', fontWeight: 500 }}>🎯 Bureau</span>
        },
      },
    ])

    ctx.register({
      id: 'palette',
      area: PALETTE_AREA,
      data: {
        id: 'focus-mode.toggle',
        label: 'Activer / Désactiver le Mode Bureau',
        keywords: ['focus', 'bureau', 'zen', 'concentration', 'plein écran'],
        run: toggleFocus,
      },
    })

    ctx.register({
      id: 'keybind',
      area: KEYBINDS_AREA,
      data: {
        id: 'focus-mode.toggle',
        category: 'view',
        defaults: ['mod+shift+f'],
        label: 'Basculer le Mode Bureau',
        run: toggleFocus,
      },
    })
  },
}

export default plugin
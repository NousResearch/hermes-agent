import type { HermesPlugin } from '@hermes/plugin-sdk'
import { COMPOSER_AREAS } from '@hermes/plugin-sdk'

const ACTIONS = [
  { icon: '📧', label: 'Rédiger un email', prompt: 'Rédige un email professionnel pour [destinataire] au sujet de [sujet]. Sois concis et poli.' },
  { icon: '📄', label: 'Analyser un document', prompt: 'Analyse le document ci-joint et donne-moi un résumé structuré avec les points clés et les actions à retenir.' },
  { icon: '📊', label: 'Préparer une présentation', prompt: 'Prépare un plan de présentation détaillé sur [sujet]. Inclus une accroche, 3-5 parties et une conclusion.' },
  { icon: '📝', label: 'Résumer un texte', prompt: 'Résume ce texte en quelques phrases. Conserve les informations essentielles.' },
  { icon: '🔍', label: 'Rechercher', prompt: 'Recherche des informations fiables sur [sujet]. Fournis des sources et synthétise.' },
]

const CHIP_STYLE: React.CSSProperties = {
  display: 'inline-flex', alignItems: 'center', gap: 6,
  padding: '6px 14px', background: '#FFFFFF',
  border: '1px solid #D8D0C2', borderRadius: 8,
  fontSize: '0.8rem', fontWeight: 500, color: '#1C1A17',
  cursor: 'pointer', whiteSpace: 'nowrap',
  transition: 'transform 0.15s ease, box-shadow 0.15s ease',
  fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif',
}

const plugin: HermesPlugin = {
  id: 'quick-actions',
  name: 'Actions rapides',
  description: "Boutons d'actions rapides pour démarrer une tâche courante.",
  defaultEnabled: true,

  register(ctx) {
    ctx.registerMany([
      {
        id: 'bar',
        area: COMPOSER_AREAS.before,
        render() {
          const handleClick = (prompt: string) => {
            // Try to find the composer textarea and set its value
            const textarea = document.querySelector('textarea[placeholder*="What"]') as HTMLTextAreaElement | null
            if (textarea) {
              const nativeSetter = Object.getOwnPropertyDescriptor(
                HTMLTextAreaElement.prototype, 'value'
              )?.set
              if (nativeSetter) {
                nativeSetter.call(textarea, prompt)
              } else {
                textarea.value = prompt
              }
              textarea.dispatchEvent(new Event('input', { bubbles: true }))
              textarea.focus()
            }
          }

          return (
            <div style={{
              display: 'flex', flexWrap: 'wrap', gap: 8, padding: '8px 0',
              justifyContent: 'center',
            }}>
              {ACTIONS.map((a) => (
                <button
                  key={a.label}
                  type="button"
                  style={CHIP_STYLE}
                  onClick={() => handleClick(a.prompt)}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = 'translateY(-1px)'
                    e.currentTarget.style.boxShadow = '0 2px 8px rgba(0,0,0,0.08)'
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = ''
                    e.currentTarget.style.boxShadow = ''
                  }}
                >
                  <span>{a.icon}</span>
                  <span>{a.label}</span>
                </button>
              ))}
            </div>
          )
        },
      },
    ])
  },
}

export default plugin
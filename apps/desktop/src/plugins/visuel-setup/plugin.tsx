import { useState } from 'react'
import type { HermesPlugin } from '@hermes/plugin-sdk'
import {
  host,
  ROUTES_AREA,
  SIDEBAR_NAV_AREA,
} from '@hermes/plugin-sdk'

const THEMES = [
  { id: 'unitalk', label: 'Unitalk', color: '#D10E63', bg: '#F3EFE6' },
  { id: 'microsoft', label: 'Microsoft', color: '#0078D4', bg: '#FFFFFF' },
  { id: 'apple', label: 'Apple', color: '#007AFF', bg: '#F5F5F7' },
  { id: 'google', label: 'Google', color: '#4285F4', bg: '#FFFFFF' },
  { id: 'chatgpt', label: 'ChatGPT', color: '#10A37F', bg: '#FFFFFF' },
  { id: 'claude', label: 'Claude', color: '#D97706', bg: '#F9F7F2' },
  { id: 'grok', label: 'Grok', color: '#000000', bg: '#FFFFFF' },
  { id: 'github', label: 'GitHub', color: '#24292F', bg: '#FFFFFF' },
  { id: 'midnight', label: 'Midnight', color: '#6366F1', bg: '#0F172A' },
  { id: 'slate', label: 'Slate', color: '#64748B', bg: '#F8FAFC' },
  { id: 'forest', label: 'Forest', color: '#4D7C0F', bg: '#F7F9F3' },
]

const CARD = {
  background: '#FFFFFF',
  border: '1px solid #D8D0C2',
  borderRadius: 12,
  padding: 32,
  maxWidth: 480,
  width: '100%',
  boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
} as const

const FONT = '-apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif'
const ACCENT = '#D10E63'
const TEXT = '#1C1A17'
const MUTED = '#7A7468'

const Step1Name = ({ onNext }: { onNext: (name: string) => void }) => {
  const [name, setName] = useState('')
  return (
    <div style={{ ...CARD, textAlign: 'center' }}>
      <div style={{ fontSize: '2.5rem', marginBottom: 12 }}>🤝</div>
      <h2 style={{ fontFamily: FONT, fontSize: '1.3rem', fontWeight: 700, color: TEXT, margin: '0 0 8px' }}>
        Donnez un nom à votre Collaborateur
      </h2>
      <p style={{ fontFamily: FONT, fontSize: '0.85rem', color: MUTED, margin: '0 0 24px', lineHeight: 1.5 }}>
        Ce sera le prénom que vous utiliserez pour lui parler au quotidien.
      </p>
      <input
        type="text"
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="Ex: Sophie, Alex..."
        style={{
          width: '100%', maxWidth: 280, padding: '10px 14px',
          border: `1.5px solid ${name ? ACCENT : '#D8D0C2'}`,
          borderRadius: 8, fontSize: '1rem', fontFamily: FONT,
          color: TEXT, outline: 'none', textAlign: 'center',
          transition: 'border-color 0.2s',
        }}
      />
      <div style={{ marginTop: 24, display: 'flex', gap: 12, justifyContent: 'center' }}>
        <button onClick={() => host.navigate('/')} style={{
          padding: '10px 20px', background: 'transparent', color: MUTED,
          border: '1px solid #D8D0C2', borderRadius: 10, fontSize: '0.85rem',
          fontFamily: FONT, fontWeight: 500, cursor: 'pointer',
        }} type="button">Plus tard</button>
        <button disabled={!name.trim()} onClick={() => onNext(name.trim())} style={{
          padding: '10px 28px', background: name.trim() ? ACCENT : '#D8D0C2',
          color: '#fff', border: 'none', borderRadius: 10, fontSize: '0.85rem',
          fontFamily: FONT, fontWeight: 600, cursor: name.trim() ? 'pointer' : 'default',
        }} type="button">Continuer</button>
      </div>
    </div>
  )
}

const Step2Theme = ({ onNext, onBack }: { onNext: (theme: string) => void; onBack: () => void }) => {
  const [selected, setSelected] = useState('unitalk')
  return (
    <div style={{ ...CARD }}>
      <div style={{ fontSize: '2.5rem', marginBottom: 12, textAlign: 'center' }}>🎨</div>
      <h2 style={{ fontFamily: FONT, fontSize: '1.3rem', fontWeight: 700, color: TEXT, margin: '0 0 8px', textAlign: 'center' }}>
        Choisissez votre ambiance
      </h2>
      <p style={{ fontFamily: FONT, fontSize: '0.85rem', color: MUTED, margin: '0 0 20px', textAlign: 'center' }}>
        Vous pourrez changer de thème à tout moment.
      </p>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))', gap: 10 }}>
        {THEMES.map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => setSelected(t.id)}
            style={{
              display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6,
              padding: '14px 10px', borderRadius: 10, border: selected === t.id ? `2px solid ${ACCENT}` : '1px solid #D8D0C2',
              background: selected === t.id ? '#FDF5F8' : '#FFFFFF', cursor: 'pointer',
              transition: 'all 0.15s ease',
            }}
          >
            <div style={{ width: 24, height: 24, borderRadius: 6, background: t.color }} />
            <span style={{ fontFamily: FONT, fontSize: '0.75rem', fontWeight: 500, color: TEXT }}>{t.label}</span>
          </button>
        ))}
      </div>
      <div style={{ marginTop: 24, display: 'flex', gap: 12, justifyContent: 'center' }}>
        <button onClick={onBack} style={{
          padding: '10px 20px', background: 'transparent', color: MUTED,
          border: '1px solid #D8D0C2', borderRadius: 10, fontSize: '0.85rem',
          fontFamily: FONT, fontWeight: 500, cursor: 'pointer',
        }} type="button">← Retour</button>
        <button onClick={() => onNext(selected)} style={{
          padding: '10px 28px', background: ACCENT, color: '#fff',
          border: 'none', borderRadius: 10, fontSize: '0.85rem',
          fontFamily: FONT, fontWeight: 600, cursor: 'pointer',
        }} type="button">Continuer</button>
      </div>
    </div>
  )
}

const Step3Mission = ({ name, theme, onDone }: { name: string; theme: string; onDone: () => void }) => {
  const suggestions = [
    'Trouve 10 prospects dans le secteur de la tech à Paris',
    'Rédige un compte-rendu de réunion à partir de mes notes',
    'Analyse ce rapport financier et donne-moi les points clés',
  ]

  const launch = (prompt: string) => {
    try { localStorage.setItem('unitalk-setup-name', name) } catch {}
    try { localStorage.setItem('unitalk-setup-theme', theme) } catch {}
    try { localStorage.setItem('unitalk-setup-complete', 'true') } catch {}

    const textarea = document.querySelector('textarea[placeholder*="What"]') as HTMLTextAreaElement | null
    if (textarea) {
      const s = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')?.set
      if (s) s.call(textarea, prompt); else textarea.value = prompt
      textarea.dispatchEvent(new Event('input', { bubbles: true }))
    }
    host.navigate('/')
  }

  return (
    <div style={{ ...CARD, textAlign: 'center' }}>
      <div style={{ fontSize: '2.5rem', marginBottom: 12 }}>🚀</div>
      <h2 style={{ fontFamily: FONT, fontSize: '1.3rem', fontWeight: 700, color: TEXT, margin: '0 0 8px' }}>
        Confiez une première mission à {name}
      </h2>
      <p style={{ fontFamily: FONT, fontSize: '0.85rem', color: MUTED, margin: '0 0 20px', lineHeight: 1.5 }}>
        Décrivez ce que vous voulez accomplir. Votre Collaborateur se met au travail immédiatement.
      </p>
      {suggestions.map((s, i) => (
        <button
          key={i}
          type="button"
          onClick={() => launch(s)}
          style={{
            display: 'block', width: '100%', textAlign: 'left',
            padding: '12px 16px', margin: '0 auto 8px', maxWidth: 400,
            background: '#F9F7F3', border: '1px solid #D8D0C2', borderRadius: 10,
            fontSize: '0.82rem', fontFamily: FONT, color: TEXT, cursor: 'pointer',
            transition: 'all 0.15s ease',
          }}
          onMouseEnter={(e) => { e.currentTarget.style.background = '#F0EDE7'; e.currentTarget.style.borderColor = ACCENT }}
          onMouseLeave={(e) => { e.currentTarget.style.background = '#F9F7F3'; e.currentTarget.style.borderColor = '#D8D0C2' }}
        >
          {s}
        </button>
      ))}
      <div style={{ marginTop: 8 }}>
        <button onClick={() => launch('')} style={{
          padding: '8px 18px', background: 'transparent', color: MUTED,
          border: 'none', fontSize: '0.8rem', fontFamily: FONT, cursor: 'pointer',
        }} type="button">Passer cette étape</button>
      </div>
    </div>
  )
}

const SetupWizard = () => {
  const [step, setStep] = useState(1)
  const [name, setName] = useState('')
  const [theme, setTheme] = useState('unitalk')

  return (
    <div style={{
      minHeight: '100vh', display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center',
      background: '#F3EFE6', fontFamily: FONT, padding: 24,
    }}>
      {/* Progress dots */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 24 }}>
        {[1, 2, 3].map((s) => (
          <div key={s} style={{
            width: 10, height: 10, borderRadius: '50%',
            background: s <= step ? ACCENT : '#D8D0C2',
            transition: 'background 0.3s ease',
          }} />
        ))}
      </div>

      {step === 1 && <Step1Name onNext={(n) => { setName(n); setStep(2) }} />}
      {step === 2 && <Step2Theme onNext={(t) => { setTheme(t); setStep(3) }} onBack={() => setStep(1)} />}
      {step === 3 && <Step3Mission name={name} theme={theme} onDone={() => host.navigate('/')} />}
    </div>
  )
}

const plugin: HermesPlugin = {
  id: 'visuel-setup',
  name: 'Guide de démarrage',
  description: 'Un guide illustré en 3 étapes pour démarrer avec votre Collaborateur IA.',
  defaultEnabled: true,

  register(ctx) {
    ctx.registerMany([
      {
        id: 'page',
        area: ROUTES_AREA,
        data: { path: '/setup' },
        render: () => <SetupWizard />,
      },
      {
        id: 'nav',
        area: SIDEBAR_NAV_AREA,
        order: 1,
        data: { codicon: 'rocket', label: 'Démarrage', path: '/setup' },
      },
    ])
  },
}

export default plugin
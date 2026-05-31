import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './app.tsx'
import './styles.css'

// Default to LIGHT mode — matches the Hermes desktop's default. The
// desktop's runtime theme system can switch to .dark later, but our
// installer ships in light mode only since we don't carry the theme
// provider machinery.
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>
)

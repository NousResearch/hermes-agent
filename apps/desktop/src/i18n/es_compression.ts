import type { FieldCopyTree } from '@/app/settings/field-copy'

// Settings > Memory & Context > Context & compression copy for this locale.
export const esCompressionFieldLabels: FieldCopyTree = {
  context: {
    engine: 'Motor de contexto'
  },
  compression: {
    enabled: 'Compresión automática',
    threshold: 'Umbral de compresión',
    codexGpt55Autoraise: 'Aumento automático de compresión de Codex',
    targetRatio: 'Objetivo de compresión',
    protectLastN: 'Mensajes recientes protegidos',
    warmHandoff: 'Traspaso en caliente'
  },
  auxiliary: {
    compression: {
      timeout: 'Tiempo de espera del modelo de compresión (s)'
    }
  }
}

export const esCompressionFieldDescriptions: FieldCopyTree = {
  context: {
    engine: 'Estrategia para gestionar conversaciones largas cerca del límite de contexto.'
  },
  compression: {
    enabled: 'Resume contexto antiguo cuando las conversaciones crecen.',
    codexGpt55Autoraise: 'Sube la compresión al 85 % para los modelos compatibles de ChatGPT Codex OAuth.',
    warmHandoff:
      'El modelo principal escribe el resumen de compresión sobre su prompt en caché, así un servidor con caché de prompts solo lee los mensajes nuevos. Más rápido cuando la compresión usa el mismo modelo. Auto: solo cuando la compresión usa el modelo principal y el servidor informa tokens en caché. Activado: intentarlo siempre. Desactivado: usar siempre el modelo de compresión. Ante cualquier fallo se usa el resumen normal.'
  },
  auxiliary: {
    compression: {
      timeout:
        'Segundos de espera del modelo auxiliar de compresión por llamada (120 por defecto). Súbelo para modelos locales lentos.'
    }
  }
}

import type { TranslationOverrides } from './define-locale'

// Simple-mode surfaces (interface-mode picker, chat starters), spread into es.ts.
export const esSimpleMode = {
  interfaceMode: {
    title: 'Modo de interfaz',
    hint: 'Cambia lo que se muestra, no lo que Hermes puede hacer.',
    sessionNote:
      'Definido por el modo Simple. Un cambio aquí dura esta sesión; cambia a Avanzado para que sea permanente.',
    switchToAdvanced: 'Cambiar a Avanzado',
    simple: {
      label: 'Simple',
      description: 'Para hablar con Hermes. Barra lateral y chat; sin paneles de terminal, archivos ni diferencias.'
    },
    advanced: {
      label: 'Avanzado',
      description:
        'Para desarrolladores. Terminal, archivos, diferencias, barra de estado y diseños, como los configures.'
    }
  },
  chatStarters: {
    label: 'Ideas para empezar',
    intro: 'Pregúntame lo que quieras, o empieza con una de estas ideas.',
    email: {
      title: 'Ayúdame a escribir un correo',
      prompt: 'Ayúdame a escribir un correo. Te cuento para quién es y lo que quiero decir: '
    },
    dinners: {
      title: 'Planea las cenas de esta semana',
      prompt: 'Planea cinco cenas fáciles para esta semana y dame una sola lista de la compra para todas.'
    },
    summarize: {
      title: 'Resume un documento',
      prompt: 'Voy a adjuntar un archivo: dime los puntos clave.'
    },
    lookup: {
      title: 'Busca algo por mí',
      prompt: 'Busca esto por mí y cuéntame lo que encuentres: '
    }
  }
} satisfies Pick<TranslationOverrides, 'interfaceMode' | 'chatStarters'>

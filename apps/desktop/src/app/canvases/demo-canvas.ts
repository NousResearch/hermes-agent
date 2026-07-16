import type { CanvasDefinition } from './types'

export const DEMO_CANVAS: CanvasDefinition = {
  id: 'reporte-operativo-julio-2026',
  version: 1,
  title: 'Reporte operativo · Julio 2026',
  profile: 'daniel',
  summary: 'Resumen ejecutivo de adquisición, conversión y principales oportunidades de seguimiento.',
  createdAt: '2026-07-15T16:30:00-06:00',
  updatedAt: '2026-07-15T16:30:00-06:00',
  source: {
    prompt: 'Crea un reporte ejecutivo mensual de adquisición y conversión.',
    instructions:
      'Actualizar mensualmente con las métricas del periodo y conservar la comparación contra el mes anterior.'
  },
  blocks: [
    {
      type: 'kpis',
      items: [
        { label: 'Leads calificados', value: '248', change: '+18.4%' },
        { label: 'Tasa de conversión', value: '12.6%', change: '+2.1 pp' },
        { label: 'Ingresos atribuidos', value: '$386,400', change: '+9.8%' }
      ]
    },
    {
      type: 'bar-chart',
      id: 'leads-por-canal',
      title: 'Leads calificados por canal',
      data: [
        { label: 'Búsqueda', value: 96, color: '#ff385f' },
        { label: 'Referidos', value: 58, color: '#ff6b84' },
        { label: 'Meta', value: 43, color: '#ff96a8' },
        { label: 'Orgánico', value: 31, color: '#ffc2cc' },
        { label: 'Eventos', value: 20, color: '#ffe0e5' }
      ]
    },
    {
      type: 'pie-chart',
      id: 'embudo',
      title: 'Distribución por etapa',
      data: [
        { label: 'Nuevos', value: 52, color: '#ff385f' },
        { label: 'En conversación', value: 28, color: '#ff8da2' },
        { label: 'Propuesta', value: 13, color: '#ffc2cc' },
        { label: 'Cerrados', value: 7, color: '#5c667a' }
      ]
    },
    {
      type: 'insight',
      id: 'insight-principal',
      title: 'Lectura principal',
      body: 'Búsqueda generó el 39% de los leads calificados y mantuvo la mejor conversión. Conviene incrementar su presupuesto antes de ampliar Meta, que todavía requiere optimización creativa.'
    },
    {
      type: 'table',
      id: 'oportunidades',
      title: 'Oportunidades prioritarias',
      columns: ['Cuenta', 'Etapa', 'Valor estimado', 'Siguiente acción'],
      rows: [
        ['Grupo Alameda', 'Propuesta', '$72,000', 'Revisión comercial · 17 jul'],
        ['Clínica Norte', 'Negociación', '$58,000', 'Enviar alcance final · 16 jul'],
        ['Delta Consultores', 'Descubrimiento', '$41,000', 'Agendar demo · 18 jul']
      ]
    }
  ]
}

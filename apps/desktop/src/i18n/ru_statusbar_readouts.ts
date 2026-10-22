// Per-turn status-bar readout copy (попадания в кэш / пропускная способность / расход сеанса).
// Split out of ru.ts, which the code-health line cap may only reduce.
import type { StatusbarReadoutTranslations } from './types_statusbar_readouts'

export const ruStatusbarReadouts: StatusbarReadoutTranslations = {
  cacheHitRateTitle: 'Доля попаданий в кэш промпта за сеанс — кэшированные токены дешевле, чем выше, тем дешевле',
  sessionCostTitle: 'Расход на этот сеанс. Префикс ~ означает нашу оценку по ценам провайдера — итоговый счёт может отличаться; $0.00 означает, что вызовы покрыты подпиской',
  tokensPerSecondTitle: 'Выходных токенов в секунду, среднее за последние 10 вызовов модели',
  toggleCacheHitRate: 'Попадания в кэш',
  toggleSessionCost: 'Стоимость сеанса',
  toggleTokensPerSecond: 'Токенов в секунду'
}

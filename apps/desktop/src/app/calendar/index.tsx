import { useStore } from '@nanostores/react'
import type * as React from 'react'
import { useCallback, useEffect, useMemo, useState } from 'react'

import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import type { CalendarDay, CalendarMonthResponse, DailyReportResponse } from '@/hermes'
import { getCalendarMonth, getDailyReport } from '@/hermes'

const MONTHS = [
  'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
]
const WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

export function CalendarView() {
  const now = new Date()
  const [year, setYear] = useState(now.getFullYear())
  const [month, setMonth] = useState(now.getMonth() + 1)
  const [loading, setLoading] = useState(true)
  const [calendarData, setCalendarData] = useState<CalendarMonthResponse | null>(null)
  const [selectedDay, setSelectedDay] = useState<CalendarDay | null>(null)

  const loadMonth = useCallback(async () => {
    setLoading(true)
    setSelectedDay(null)
    try {
      const data = await getCalendarMonth(year, month)
      setCalendarData(data)
    } catch {
      setCalendarData(null)
    }
    setLoading(false)
  }, [year, month])

  useEffect(() => {
    void loadMonth()
  }, [loadMonth])

  const prevMonth = useCallback(() => {
    if (month === 1) {
      setYear(y => y - 1)
      setMonth(12)
    } else {
      setMonth(m => m - 1)
    }
  }, [month])

  const nextMonth = useCallback(() => {
    if (month === 12) {
      setYear(y => y + 1)
      setMonth(1)
    } else {
      setMonth(m => m + 1)
    }
  }, [month])

  // First day of month (0=Sun, adjust to Mon=0)
  const firstDay = useMemo(() => {
    const d = new Date(year, month - 1, 1)
    const dw = d.getDay()
    return dw === 0 ? 6 : dw - 1  // Mon=0 ... Sun=6
  }, [year, month])

  const daysInMonth = useMemo(() => {
    return new Date(year, month, 0).getDate()
  }, [year, month])

  // Build grid: leading blanks + day cells + trailing blanks
  const gridCells: (number | null)[] = useMemo(() => {
    const cells: (number | null)[] = Array(firstDay).fill(null)
    for (let d = 1; d <= daysInMonth; d++) {
      cells.push(d)
    }
    while (cells.length % 7 !== 0) {
      cells.push(null)
    }
    return cells
  }, [firstDay, daysInMonth])

  const dayLookup = useMemo(() => {
    const map = new Map<number, CalendarDay>()
    if (calendarData) {
      for (const d of calendarData.days) {
        map.set(d.day, d)
      }
    }
    return map
  }, [calendarData])

  const activityCount = calendarData?.activityCount ?? 0

  return (
    <div className="flex h-full flex-col overflow-hidden">
      {/* Header */}
      <div className="flex shrink-0 items-center justify-between border-b px-6 py-3">
        <h1 className="text-lg font-semibold">Activity Calendar</h1>
        <span className="text-sm text-(--ui-text-tertiary)">
          {activityCount} day{activityCount !== 1 ? 's' : ''} with activity this month
        </span>
      </div>

      <div className="flex flex-1 flex-col overflow-auto p-6">
        {/* Month navigation */}
        <div className="mb-4 flex items-center justify-between">
          <Button onClick={prevMonth} size="icon" variant="ghost">
            <Codicon name="chevron-left" size="1rem" />
          </Button>
          <span className="text-base font-medium">
            {MONTHS[month - 1]} {year}
          </span>
          <Button onClick={nextMonth} size="icon" variant="ghost">
            <Codicon name="chevron-right" size="1rem" />
          </Button>
        </div>

        {loading ? (
          <PageLoader />
        ) : (
          <>
            {/* Weekday headers */}
            <div className="mb-1 grid grid-cols-7 gap-px">
              {WEEKDAYS.map(wd => (
                <div key={wd} className="py-1 text-center text-xs font-medium text-(--ui-text-tertiary)">
                  {wd}
                </div>
              ))}
            </div>

            {/* Day grid */}
            <div className="grid grid-cols-7 gap-px">
              {gridCells.map((day, i) => {
                if (day === null) {
                  return <div key={`empty-${i}`} className="aspect-square" />
                }
                const info = dayLookup.get(day)
                const hasActivity = info?.hasActivity ?? false
                const isToday =
                  day === now.getDate() &&
                  month === now.getMonth() + 1 &&
                  year === now.getFullYear()
                const isSelected = selectedDay?.day === day

                return (
                  <Tip key={day} label={info?.preview || (hasActivity ? 'has activity' : 'no activity')}>
                    <button
                      className={cn(
                        'relative flex aspect-square flex-col items-center justify-center rounded-md text-sm transition-colors',
                        isSelected && 'bg-(--ui-row-active-background) font-semibold',
                        !isSelected && hasActivity && 'hover:bg-(--chrome-action-hover)',
                        !isSelected && !hasActivity && 'text-(--ui-text-quaternary)',
                        isToday && !isSelected && 'ring-1 ring-(--ui-border)',
                      )}
                      type="button"
                    >
                      <span>{day}</span>
                      {hasActivity && (
                        <span className="mt-0.5 size-1 rounded-full bg-(--ui-accent)" />
                      )}
                    </button>
                  </Tip>
                )
              })}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
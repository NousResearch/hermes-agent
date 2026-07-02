// The _PyTime_t API is written to use timestamp and timeout values stored in
// various formats and to read clocks.
//
// The _PyTime_t type is an integer to support directly common arithmetic
// operations like t1 + t2.
//
// The _PyTime_t API supports a resolution of 1 nanosecond. The _PyTime_t type
// is signed to support negative timestamps. The supported range is around
// [-292.3 years; +292.3 years]. Using the Unix epoch (January 1st, 1970), the
// supported date range is around [1677-09-21; 2262-04-11].
//
// Formats:
//
// * seconds
// * seconds as a floating pointer number (C double)
// * milliseconds (10^-3 seconds)
// * microseconds (10^-6 seconds)
// * 100 nanoseconds (10^-7 seconds)
// * nanoseconds (10^-9 seconds)
// * timeval structure, 1 microsecond resolution (10^-6 seconds)
// * timespec structure, 1 nanosecond resolution (10^-9 seconds)
//
// Integer overflows are detected and raise OverflowError. Conversion to a
// resolution worse than 1 nanosecond is rounded correctly with the requested
// rounding mode. There are 4 rounding modes: floor (towards -inf), ceiling
// (towards +inf), half even and up (away from zero).
//
// Some functions clamp the result in the range [_PyTime_MIN; _PyTime_MAX], so
// the caller doesn't have to handle errors and doesn't need to hold the GIL.
// For example, _PyTime_Add(t1, t2) computes t1+t2 and clamp the result on
// overflow.
//
// Clocks:
//
// * System clock
// * Monotonic clock
// * Performance counter
//
// Operations like (t * k / q) with integers are implemented in a way to reduce
// the risk of integer overflow. Such operation is used to convert a clock
// value expressed in ticks with a frequency to _PyTime_t, like
// QueryPerformanceCounter() with QueryPerformanceFrequency().

#ifndef Py_LIMITED_API
#ifndef Py_PYTIME_H
#define Py_PYTIME_H

/**************************************************************************
Symbols and macros to supply platform-independent interfaces to time related
functions and constants
**************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif

/* _PyTime_t: Python timestamp with subsecond precision. It can be used to
   store a duration, and so indirectly a date (related to another date, like
   UNIX epoch). */
typedef int64_t _PyTime_t;
// _PyTime_MIN nanoseconds is around -292.3 years
#define _PyTime_MIN INT64_MIN
// _PyTime_MAX nanoseconds is around +292.3 years
#define _PyTime_MAX INT64_MAX
#define _SIZEOF_PYTIME_T 8

typedef enum {
    /* Round towards minus infinity (-inf).
       For example, used to read a clock. */
    _PyTime_ROUND_FLOOR=0,
    /* Round towards infinity (+inf).
       For example, used for timeout to wait "at least" N seconds. */
    _PyTime_ROUND_CEILING=1,
    /* Round to nearest with ties going to nearest even integer.
       For example, used to round from a Python float. */
    _PyTime_ROUND_HALF_EVEN=2,
    /* Round away from zero
       For example, used for timeout. _PyTime_ROUND_CEILING rounds
       -1e-9 to 0 milliseconds which causes bpo-31786 issue.
       _PyTime_ROUND_UP rounds -1e-9 to -1 millisecond which keeps
       the timeout sign as expected. select.poll(timeout) must block
       for negative values." */
    _PyTime_ROUND_UP=3,
    /* _PyTime_ROUND_TIMEOUT (an alias for _PyTime_ROUND_UP) should be
       used for timeouts. */
    _PyTime_ROUND_TIMEOUT = _PyTime_ROUND_UP
} _PyTime_round_t;


/* Convert a time_t to a PyLong. */
PyAPI_FUNC(PyObject *) _PyLong_FromTime_t(
    time_t sec);

/* Convert a PyLong to a time_t. */
PyAPI_FUNC(time_t) _PyLong_AsTime_t(
    PyObject *obj);

/* Convert a number of seconds, int or float, to time_t. */
PyAPI_FUNC(int) _PyTime_ObjectToTime_t(
    PyObject *obj,
    time_t *sec,
    _PyTime_round_t);

/* Convert a number of seconds, int or float, to a timeval structure.
   usec is in the range [0; 999999] and rounded towards zero.
   For example, -1.2 is converted to (-2, 800000). */
PyAPI_FUNC(int) _PyTime_ObjectToTimeval(
    PyObject *obj,
    time_t *sec,
    long *usec,
    _PyTime_round_t);

/* Convert a number of seconds, int or float, to a timespec structure.
   nsec is in the range [0; 999999999] and rounded towards zero.
   For example, -1.2 is converted to (-2, 800000000). */
PyAPI_FUNC(int) _PyTime_ObjectToTimespec(
    PyObject *obj,
    time_t *sec,
    long *nsec,
    _PyTime_round_t);


/* Create a timestamp from a number of seconds. */
PyAPI_FUNC(_PyTime_t) _PyTime_FromSeconds(int seconds);

/* Macro to create a timestamp from a number of seconds, no integer overflow.
   Only use the macro for small values, prefer _PyTime_FromSeconds(). */
#define _PYTIME_FROMSECONDS(seconds) \
            ((_PyTime_t)(seconds) * (1000 * 1000 * 1000))

/* Create a timestamp from a number of nanoseconds. */
PyAPI_FUNC(_PyTime_t) _PyTime_FromNanoseconds(_PyTime_t ns);

/* Create a timestamp from nanoseconds (Python int). */
PyAPI_FUNC(int) _PyTime_FromNanosecondsObject(_PyTime_t *t,
    PyObject *obj);

/* Convert a number of seconds (Python float or int) to a timestamp.
   Raise an exception and return -1 on error, return 0 on success. */
PyAPI_FUNC(int) _PyTime_FromSecondsObject(_PyTime_t *t,
    PyObject *obj,
    _PyTime_round_t round);

/* Convert a number of milliseconds (Python float or int, 10^-3) to a timestamp.
   Raise an exception and return -1 on error, return 0 on success. */
PyAPI_FUNC(int) _PyTime_FromMillisecondsObject(_PyTime_t *t,
    PyObject *obj,
    _PyTime_round_t round);

/* Convert a timestamp to a number of seconds as a C double. */
PyAPI_FUNC(double) _PyTime_AsSecondsDouble(_PyTime_t t);

/* Convert timestamp to a number of milliseconds (10^-3 seconds). */
PyAPI_FUNC(_PyTime_t) _PyTime_AsMilliseconds(_PyTime_t t,
    _PyTime_round_t round);

/* Convert timestamp to a number of microseconds (10^-6 seconds). */
PyAPI_FUNC(_PyTime_t) _PyTime_AsMicroseconds(_PyTime_t t,
    _PyTime_round_t round);

/* Convert timestamp to a number of nanoseconds (10^-9 seconds). */
PyAPI_FUNC(_PyTime_t) _PyTime_AsNanoseconds(_PyTime_t t);

#ifdef MS_WINDOWS
// Convert timestamp to a number of 100 nanoseconds (10^-7 seconds).
PyAPI_FUNC(_PyTime_t) _PyTime_As100Nanoseconds(_PyTime_t t,
    _PyTime_round_t round);
#endif

/* Convert timestamp to a number of nanoseconds (10^-9 seconds) as a Python int
   object. */
PyAPI_FUNC(PyObject *) _PyTime_AsNanosecondsObject(_PyTime_t t);

#ifndef MS_WINDOWS
/* Create a timestamp from a timeval structure.
   Raise an exception and return -1 on overflow, return 0 on success. */
PyAPI_FUNC(int) _PyTime_FromTimeval(_PyTime_t *tp, struct timeval *tv);
#endif

/* Convert a timestamp to a timeval structure (microsecond resolution).
   tv_usec is always positive.
   Raise an exception and return -1 if the conversion overflowed,
   return 0 on success. */
PyAPI_FUNC(int) _PyTime_AsTimeval(_PyTime_t t,
    struct timeval *tv,
    _PyTime_round_t round);

/* Similar to _PyTime_AsTimeval() but don't raise an exception on overflow.
   On overflow, clamp tv_sec to _PyTime_t min/max. */
PyAPI_FUNC(void) _PyTime_AsTimeval_clamp(_PyTime_t t,
    struct timeval *tv,
    _PyTime_round_t round);

/* Convert a timestamp to a number of seconds (secs) and microseconds (us).
   us is always positive. This function is similar to _PyTime_AsTimeval()
   except that secs is always a time_t type, whereas the timeval structure
   uses a C long for tv_sec on Windows.
   Raise an exception and return -1 if the conversion overflowed,
   return 0 on success. */
PyAPI_FUNC(int) _PyTime_AsTimevalTime_t(
    _PyTime_t t,
    time_t *secs,
    int *us,
    _PyTime_round_t round);

#if defined(HAVE_CLOCK_GETTIME) || defined(HAVE_KQUEUE)
/* Create a timestamp from a timespec structure.
   Raise an exception and return -1 on overflow, return 0 on success. */
PyAPI_FUNC(int) _PyTime_FromTimespec(_PyTime_t *tp, struct timespec *ts);

/* Convert a timestamp to a timespec structure (nanosecond resolution).
   tv_nsec is always positive.
   Raise an exception and return -1 on error, return 0 on success. */
PyAPI_FUNC(int) _PyTime_AsTimespec(_PyTime_t t, struct timespec *ts);

/* Similar to _PyTime_AsTimespec() but don't raise an exception on overflow.
   On overflow, clamp tv_sec to _PyTime_t min/max. */
PyAPI_FUNC(void) _PyTime_AsTimespec_clamp(_PyTime_t t, struct timespec *ts);
#endif


// Compute t1 + t2. Clamp to [_PyTime_MIN; _PyTime_MAX] on overflow.
PyAPI_FUNC(_PyTime_t) _PyTime_Add(_PyTime_t t1, _PyTime_t t2);

/* Compute ticks * mul / div.
   Clamp to [_PyTime_MIN; _PyTime_MAX] on overflow.
   The caller must ensure that ((div - 1) * mul) cannot overflow. */
PyAPI_FUNC(_PyTime_t) _PyTime_MulDiv(_PyTime_t ticks,
    _PyTime_t mul,
    _PyTime_t div);

/* Structure used by time.get_clock_info() */
typedef struct {
    const char *implementation;
    int monotonic;
    int adjustable;
    double resolution;
} _Py_clock_info_t;

/* Get the current time from the system clock.

   If the internal clock fails, silently ignore the error and return 0.
   On integer overflow, silently ignore the overflow and clamp the clock to
   [_PyTime_MIN; _PyTime_MAX].

   Use _PyTime_GetSystemClockWithInfo() to check for failure. */
PyAPI_FUNC(_PyTime_t) _PyTime_GetSystemClock(void);

/* Get the current time from the system clock.
 * On success, set *t and *info (if not NULL), and return 0.
 * On error, raise an exception and return -1.
 */
PyAPI_FUNC(int) _PyTime_GetSystemClockWithInfo(
    _PyTime_t *t,
    _Py_clock_info_t *info);

/* Get the time of a monotonic clock, i.e. a clock that cannot go backwards.
   The clock is not affected by system clock updates. The reference point of
   the returned value is undefined, so that only the difference between the
   results of consecutive calls is valid.

   If the internal clock fails, silently ignore the error and return 0.
   On integer overflow, silently ignore the overflow and clamp the clock to
   [_PyTime_MIN; _PyTime_MAX].

   Use _PyTime_GetMonotonicClockWithInfo() to check for failure. */
PyAPI_FUNC(_PyTime_t) _PyTime_GetMonotonicClock(void);

/* Get the time of a monotonic clock, i.e. a clock that cannot go backwards.
   The clock is not affected by system clock updates. The reference point of
   the returned value is undefined, so that only the difference between the
   results of consecutive calls is valid.

   Fill info (if set) with information of the function used to get the time.

   Return 0 on success, raise an exception and return -1 on error. */
PyAPI_FUNC(int) _PyTime_GetMonotonicClockWithInfo(
    _PyTime_t *t,
    _Py_clock_info_t *info);


/* Converts a timestamp to the Gregorian time, using the local time zone.
   Return 0 on success, raise an exception and return -1 on error. */
PyAPI_FUNC(int) _PyTime_localtime(time_t t, struct tm *tm);

/* Converts a timestamp to the Gregorian time, assuming UTC.
   Return 0 on success, raise an exception and return -1 on error. */
PyAPI_FUNC(int) _PyTime_gmtime(time_t t, struct tm *tm);

/* Get the performance counter: clock with the highest available resolution to
   measure a short duration.

   If the internal clock fails, silently ignore the error and return 0.
   On integer overflow, silently ignore the overflow and clamp the clock to
   [_PyTime_MIN; _PyTime_MAX].

   Use _PyTime_GetPerfCounterWithInfo() to check for failure. */
PyAPI_FUNC(_PyTime_t) _PyTime_GetPerfCounter(void);

/* Get the performance counter: clock with the highest available resolution to
   measure a short duration.

   Fill info (if set) with information of the function used to get the time.

   Return 0 on success, raise an exception and return -1 on error. */
PyAPI_FUNC(int) _PyTime_GetPerfCounterWithInfo(
    _PyTime_t *t,
    _Py_clock_info_t *info);


// Create a deadline.
// Pseudo code: _PyTime_GetMonotonicClock() + timeout.
PyAPI_FUNC(_PyTime_t) _PyDeadline_Init(_PyTime_t timeout);

// Get remaining time from a deadline.
// Pseudo code: deadline - _PyTime_GetMonotonicClock().
PyAPI_FUNC(_PyTime_t) _PyDeadline_Get(_PyTime_t deadline);

#ifdef __cplusplus
}
#endif

#endif /* Py_PYTIME_H */
#endif /* Py_LIMITED_API */

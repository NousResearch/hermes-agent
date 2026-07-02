#ifndef Py_INTERNAL_CONDVAR_H
#define Py_INTERNAL_CONDVAR_H

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

#ifndef _POSIX_THREADS
/* This means pthreads are not implemented in libc headers, hence the macro
   not present in unistd.h. But they still can be implemented as an external
   library (e.g. gnu pth in pthread emulation) */
# ifdef HAVE_PTHREAD_H
#  include <pthread.h> /* _POSIX_THREADS */
# endif
#endif

#ifdef _POSIX_THREADS
/*
 * POSIX support
 */
#define Py_HAVE_CONDVAR

#ifdef HAVE_PTHREAD_H
#  include <pthread.h>
#endif

#define PyMUTEX_T pthread_mutex_t
#define PyCOND_T pthread_cond_t

#elif defined(NT_THREADS)
/*
 * Windows (XP, 2003 server and later, as well as (hopefully) CE) support
 *
 * Emulated condition variables ones that work with XP and later, plus
 * example native support on VISTA and onwards.
 */
#define Py_HAVE_CONDVAR

/* include windows if it hasn't been done before */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

/* options */
/* non-emulated condition variables are provided for those that want
 * to target Windows Vista.  Modify this macro to enable them.
 */
#ifndef _PY_EMULATED_WIN_CV
#define _PY_EMULATED_WIN_CV 1  /* use emulated condition variables */
#endif

/* fall back to emulation if not targeting Vista */
#if !defined NTDDI_VISTA || NTDDI_VERSION < NTDDI_VISTA
#undef _PY_EMULATED_WIN_CV
#define _PY_EMULATED_WIN_CV 1
#endif

#if _PY_EMULATED_WIN_CV

typedef CRITICAL_SECTION PyMUTEX_T;

/* The ConditionVariable object.  From XP onwards it is easily emulated
   with a Semaphore.
   Semaphores are available on Windows XP (2003 server) and later.
   We use a Semaphore rather than an auto-reset event, because although
   an auto-reset event might appear to solve the lost-wakeup bug (race
   condition between releasing the outer lock and waiting) because it
   maintains state even though a wait hasn't happened, there is still
   a lost wakeup problem if more than one thread are interrupted in the
   critical place.  A semaphore solves that, because its state is
   counted, not Boolean.
   Because it is ok to signal a condition variable with no one
   waiting, we need to keep track of the number of
   waiting threads.  Otherwise, the semaphore's state could rise
   without bound.  This also helps reduce the number of "spurious wakeups"
   that would otherwise happen.
 */

typedef struct _PyCOND_T
{
    HANDLE sem;
    int waiting; /* to allow PyCOND_SIGNAL to be a no-op */
} PyCOND_T;

#else /* !_PY_EMULATED_WIN_CV */

/* Use native Win7 primitives if build target is Win7 or higher */

/* SRWLOCK is faster and better than CriticalSection */
typedef SRWLOCK PyMUTEX_T;

typedef CONDITION_VARIABLE  PyCOND_T;

#endif /* _PY_EMULATED_WIN_CV */

#endif /* _POSIX_THREADS, NT_THREADS */

#endif /* Py_INTERNAL_CONDVAR_H */

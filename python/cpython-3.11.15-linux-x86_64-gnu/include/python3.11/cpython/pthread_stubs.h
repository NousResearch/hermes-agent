#ifndef Py_CPYTHON_PTRHEAD_STUBS_H
#define Py_CPYTHON_PTRHEAD_STUBS_H

#if !defined(HAVE_PTHREAD_STUBS)
#  error "this header file requires stubbed pthreads."
#endif

#ifndef _POSIX_THREADS
#  define _POSIX_THREADS 1
#endif

/* Minimal pthread stubs for CPython.
 *
 * The stubs implement the minimum pthread API for CPython.
 * - pthread_create() fails.
 * - pthread_exit() calls exit(0).
 * - pthread_key_*() functions implement minimal TSS without destructor.
 * - all other functions do nothing and return 0.
 */

#ifdef __wasi__
// WASI's bits/alltypes.h provides type definitions when __NEED_ is set.
// The header file can be included multiple times.
#  define __NEED_pthread_cond_t 1
#  define __NEED_pthread_condattr_t 1
#  define __NEED_pthread_mutex_t 1
#  define __NEED_pthread_mutexattr_t 1
#  define __NEED_pthread_key_t 1
#  define __NEED_pthread_t 1
#  define __NEED_pthread_attr_t 1
#  include <bits/alltypes.h>
#else
typedef struct { void *__x; } pthread_cond_t;
typedef struct { unsigned __attr; } pthread_condattr_t;
typedef struct { void *__x; } pthread_mutex_t;
typedef struct { unsigned __attr; } pthread_mutexattr_t;
typedef unsigned pthread_key_t;
typedef unsigned pthread_t;
typedef struct { unsigned __attr; } pthread_attr_t;
#endif

// mutex
PyAPI_FUNC(int) pthread_mutex_init(pthread_mutex_t *restrict mutex,
                                   const pthread_mutexattr_t *restrict attr);
PyAPI_FUNC(int) pthread_mutex_destroy(pthread_mutex_t *mutex);
PyAPI_FUNC(int) pthread_mutex_trylock(pthread_mutex_t *mutex);
PyAPI_FUNC(int) pthread_mutex_lock(pthread_mutex_t *mutex);
PyAPI_FUNC(int) pthread_mutex_unlock(pthread_mutex_t *mutex);

// condition
PyAPI_FUNC(int) pthread_cond_init(pthread_cond_t *restrict cond,
                                  const pthread_condattr_t *restrict attr);
PyAPI_FUNC(int) pthread_cond_destroy(pthread_cond_t *cond);
PyAPI_FUNC(int) pthread_cond_wait(pthread_cond_t *restrict cond,
                                  pthread_mutex_t *restrict mutex);
PyAPI_FUNC(int) pthread_cond_timedwait(pthread_cond_t *restrict cond,
                                       pthread_mutex_t *restrict mutex,
                                       const struct timespec *restrict abstime);
PyAPI_FUNC(int) pthread_cond_signal(pthread_cond_t *cond);
PyAPI_FUNC(int) pthread_condattr_init(pthread_condattr_t *attr);
PyAPI_FUNC(int) pthread_condattr_setclock(
    pthread_condattr_t *attr, clockid_t clock_id);

// pthread
PyAPI_FUNC(int) pthread_create(pthread_t *restrict thread,
                               const pthread_attr_t *restrict attr,
                               void *(*start_routine)(void *),
                               void *restrict arg);
PyAPI_FUNC(int) pthread_detach(pthread_t thread);
PyAPI_FUNC(pthread_t) pthread_self(void);
PyAPI_FUNC(int) pthread_exit(void *retval) __attribute__ ((__noreturn__));
PyAPI_FUNC(int) pthread_attr_init(pthread_attr_t *attr);
PyAPI_FUNC(int) pthread_attr_setstacksize(pthread_attr_t *attr, size_t stacksize);
PyAPI_FUNC(int) pthread_attr_destroy(pthread_attr_t *attr);


// pthread_key
#ifndef PTHREAD_KEYS_MAX
#  define PTHREAD_KEYS_MAX 128
#endif

PyAPI_FUNC(int) pthread_key_create(pthread_key_t *key,
                                   void (*destr_function)(void *));
PyAPI_FUNC(int) pthread_key_delete(pthread_key_t key);
PyAPI_FUNC(void *) pthread_getspecific(pthread_key_t key);
PyAPI_FUNC(int) pthread_setspecific(pthread_key_t key, const void *value);

#endif // Py_CPYTHON_PTRHEAD_STUBS_H

/* Atomic functions: similar to pycore_atomic.h, but don't need
   to declare variables as atomic.

   Py_ssize_t type:

   * value = _Py_atomic_size_get(&var)
   * _Py_atomic_size_set(&var, value)

   Use sequentially-consistent ordering (__ATOMIC_SEQ_CST memory order):
   enforce total ordering with all other atomic functions.
*/
#ifndef Py_ATOMIC_FUNC_H
#define Py_ATOMIC_FUNC_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

#if defined(_MSC_VER)
#  include <intrin.h>             // _InterlockedExchange()
#endif


// Use builtin atomic operations in GCC >= 4.7 and clang
#ifdef HAVE_BUILTIN_ATOMIC

static inline Py_ssize_t _Py_atomic_size_get(Py_ssize_t *var)
{
    return __atomic_load_n(var, __ATOMIC_SEQ_CST);
}

static inline void _Py_atomic_size_set(Py_ssize_t *var, Py_ssize_t value)
{
    __atomic_store_n(var, value, __ATOMIC_SEQ_CST);
}

#elif defined(_MSC_VER)

static inline Py_ssize_t _Py_atomic_size_get(Py_ssize_t *var)
{
#if SIZEOF_VOID_P == 8
    Py_BUILD_ASSERT(sizeof(__int64) == sizeof(*var));
    volatile __int64 *volatile_var = (volatile __int64 *)var;
    __int64 old;
    do {
        old = *volatile_var;
    } while(_InterlockedCompareExchange64(volatile_var, old, old) != old);
#else
    Py_BUILD_ASSERT(sizeof(long) == sizeof(*var));
    volatile long *volatile_var = (volatile long *)var;
    long old;
    do {
        old = *volatile_var;
    } while(_InterlockedCompareExchange(volatile_var, old, old) != old);
#endif
    return old;
}

static inline void _Py_atomic_size_set(Py_ssize_t *var, Py_ssize_t value)
{
#if SIZEOF_VOID_P == 8
    Py_BUILD_ASSERT(sizeof(__int64) == sizeof(*var));
    volatile __int64 *volatile_var = (volatile __int64 *)var;
    _InterlockedExchange64(volatile_var, value);
#else
    Py_BUILD_ASSERT(sizeof(long) == sizeof(*var));
    volatile long *volatile_var = (volatile long *)var;
    _InterlockedExchange(volatile_var, value);
#endif
}

#else
// Fallback implementation using volatile

static inline Py_ssize_t _Py_atomic_size_get(Py_ssize_t *var)
{
    volatile Py_ssize_t *volatile_var = (volatile Py_ssize_t *)var;
    return *volatile_var;
}

static inline void _Py_atomic_size_set(Py_ssize_t *var, Py_ssize_t value)
{
    volatile Py_ssize_t *volatile_var = (volatile Py_ssize_t *)var;
    *volatile_var = value;
}
#endif

#ifdef __cplusplus
}
#endif
#endif  /* Py_ATOMIC_FUNC_H */

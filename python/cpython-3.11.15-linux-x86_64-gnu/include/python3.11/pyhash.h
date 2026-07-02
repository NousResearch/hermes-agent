#ifndef Py_HASH_H

#define Py_HASH_H
#ifdef __cplusplus
extern "C" {
#endif

/* Helpers for hash functions */
#ifndef Py_LIMITED_API
PyAPI_FUNC(Py_hash_t) _Py_HashDouble(PyObject *, double);
PyAPI_FUNC(Py_hash_t) _Py_HashPointer(const void*);
// Similar to _Py_HashPointer(), but don't replace -1 with -2
PyAPI_FUNC(Py_hash_t) _Py_HashPointerRaw(const void*);
PyAPI_FUNC(Py_hash_t) _Py_HashBytes(const void*, Py_ssize_t);
#endif

/* Prime multiplier used in string and various other hashes. */
#define _PyHASH_MULTIPLIER 1000003UL  /* 0xf4243 */

/* Parameters used for the numeric hash implementation.  See notes for
   _Py_HashDouble in Python/pyhash.c.  Numeric hashes are based on
   reduction modulo the prime 2**_PyHASH_BITS - 1. */

#if SIZEOF_VOID_P >= 8
#  define _PyHASH_BITS 61
#else
#  define _PyHASH_BITS 31
#endif

#define _PyHASH_MODULUS (((size_t)1 << _PyHASH_BITS) - 1)
#define _PyHASH_INF 314159
#define _PyHASH_IMAG _PyHASH_MULTIPLIER


/* hash secret
 *
 * memory layout on 64 bit systems
 *   cccccccc cccccccc cccccccc  uc -- unsigned char[24]
 *   pppppppp ssssssss ........  fnv -- two Py_hash_t
 *   k0k0k0k0 k1k1k1k1 ........  siphash -- two uint64_t
 *   ........ ........ ssssssss  djbx33a -- 16 bytes padding + one Py_hash_t
 *   ........ ........ eeeeeeee  pyexpat XML hash salt
 *
 * memory layout on 32 bit systems
 *   cccccccc cccccccc cccccccc  uc
 *   ppppssss ........ ........  fnv -- two Py_hash_t
 *   k0k0k0k0 k1k1k1k1 ........  siphash -- two uint64_t (*)
 *   ........ ........ ssss....  djbx33a -- 16 bytes padding + one Py_hash_t
 *   ........ ........ eeee....  pyexpat XML hash salt
 *
 * (*) The siphash member may not be available on 32 bit platforms without
 *     an unsigned int64 data type.
 */
#ifndef Py_LIMITED_API
typedef union {
    /* ensure 24 bytes */
    unsigned char uc[24];
    /* two Py_hash_t for FNV */
    struct {
        Py_hash_t prefix;
        Py_hash_t suffix;
    } fnv;
    /* two uint64 for SipHash24 */
    struct {
        uint64_t k0;
        uint64_t k1;
    } siphash;
    /* a different (!) Py_hash_t for small string optimization */
    struct {
        unsigned char padding[16];
        Py_hash_t suffix;
    } djbx33a;
    struct {
        unsigned char padding[16];
        Py_hash_t hashsalt;
    } expat;
} _Py_HashSecret_t;
PyAPI_DATA(_Py_HashSecret_t) _Py_HashSecret;

#ifdef Py_DEBUG
PyAPI_DATA(int) _Py_HashSecret_Initialized;
#endif


/* hash function definition */
typedef struct {
    Py_hash_t (*const hash)(const void *, Py_ssize_t);
    const char *name;
    const int hash_bits;
    const int seed_bits;
} PyHash_FuncDef;

PyAPI_FUNC(PyHash_FuncDef*) PyHash_GetFuncDef(void);
#endif


/* cutoff for small string DJBX33A optimization in range [1, cutoff).
 *
 * About 50% of the strings in a typical Python application are smaller than
 * 6 to 7 chars. However DJBX33A is vulnerable to hash collision attacks.
 * NEVER use DJBX33A for long strings!
 *
 * A Py_HASH_CUTOFF of 0 disables small string optimization. 32 bit platforms
 * should use a smaller cutoff because it is easier to create colliding
 * strings. A cutoff of 7 on 64bit platforms and 5 on 32bit platforms should
 * provide a decent safety margin.
 */
#ifndef Py_HASH_CUTOFF
#  define Py_HASH_CUTOFF 0
#elif (Py_HASH_CUTOFF > 7 || Py_HASH_CUTOFF < 0)
#  error Py_HASH_CUTOFF must in range 0...7.
#endif /* Py_HASH_CUTOFF */


/* hash algorithm selection
 *
 * The values for Py_HASH_* are hard-coded in the
 * configure script.
 *
 * - FNV and SIPHASH* are available on all platforms and architectures.
 * - With EXTERNAL embedders can provide an alternative implementation with::
 *
 *     PyHash_FuncDef PyHash_Func = {...};
 *
 * XXX: Figure out __declspec() for extern PyHash_FuncDef.
 */
#define Py_HASH_EXTERNAL 0
#define Py_HASH_SIPHASH24 1
#define Py_HASH_FNV 2
#define Py_HASH_SIPHASH13 3

#ifndef Py_HASH_ALGORITHM
#  ifndef HAVE_ALIGNED_REQUIRED
#    define Py_HASH_ALGORITHM Py_HASH_SIPHASH13
#  else
#    define Py_HASH_ALGORITHM Py_HASH_FNV
#  endif /* uint64_t && uint32_t && aligned */
#endif /* Py_HASH_ALGORITHM */

#ifdef __cplusplus
}
#endif

#endif /* !Py_HASH_H */

#ifndef Py_INTERNAL_HASH_H
#define Py_INTERNAL_HASH_H

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

uint64_t _Py_KeyedHash(uint64_t, const char *, Py_ssize_t);

#endif
